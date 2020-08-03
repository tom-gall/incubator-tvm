/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file rpc_module.cc
 * \brief RPC runtime module.
 */
#include <tvm/runtime/container.h>
#include <tvm/runtime/registry.h>

#include <cstring>
#include <memory>

#include "rpc_endpoint.h"
#include "rpc_session.h"

namespace tvm {
namespace runtime {

/*!
 * \brief A wrapped remote function as a PackedFunc.
 */
class RPCWrappedFunc : public Object {
 public:
  RPCWrappedFunc(void* handle, std::shared_ptr<RPCSession> sess) : handle_(handle), sess_(sess) {}

  void operator()(TVMArgs args, TVMRetValue* rv) const {
    std::vector<TVMValue> values(args.values, args.values + args.size());
    std::vector<int> type_codes(args.type_codes, args.type_codes + args.size());
    std::vector<std::unique_ptr<DLTensor>> temp_dltensors;

    // scan and check whether we need rewrite these arguments
    // to their remote variant.
    for (int i = 0; i < args.size(); ++i) {
      if (args[i].IsObjectRef<String>()) {
        String str = args[i];
        type_codes[i] = kTVMStr;
        values[i].v_str = str.c_str();
        continue;
      }
      int tcode = type_codes[i];
      switch (tcode) {
        case kTVMDLTensorHandle:
        case kTVMNDArrayHandle: {
          // Pass NDArray as DLTensor, NDArray and DLTensor
          // are compatible to each other, just need to change the index.
          type_codes[i] = kTVMDLTensorHandle;
          // translate to a remote view of DLTensor
          auto dptr = std::make_unique<DLTensor>(*static_cast<DLTensor*>(values[i].v_handle));
          dptr->ctx = RemoveSessMask(dptr->ctx);
          dptr->data = static_cast<RemoteSpace*>(dptr->data)->data;
          values[i].v_handle = dptr.get();
          temp_dltensors.emplace_back(std::move(dptr));
          break;
        }
        case kTVMContext: {
          values[i].v_ctx = RemoveSessMask(values[i].v_ctx);
          break;
        }
        case kTVMPackedFuncHandle:
        case kTVMModuleHandle: {
          values[i].v_handle = UnwrapRemoteValueToHandle(TVMArgValue(values[i], tcode));
          break;
        }
      }
    }
    auto set_return = [this, rv](TVMArgs args) { this->WrapRemoteReturnToValue(args, rv); };
    sess_->CallFunc(handle_, values.data(), type_codes.data(), args.size(), set_return);
  }

  ~RPCWrappedFunc() {
    try {
      sess_->FreeHandle(handle_, kTVMPackedFuncHandle);
    } catch (const dmlc::Error& e) {
      // fault tolerance to remote close
    }
  }

 private:
  // remote function handle
  void* handle_{nullptr};
  // pointer to the session.
  std::shared_ptr<RPCSession> sess_;

  // unwrap a remote value to the underlying handle.
  void* UnwrapRemoteValueToHandle(const TVMArgValue& arg) const;
  // wrap a remote return via Set
  void WrapRemoteReturnToValue(TVMArgs args, TVMRetValue* rv) const;

  // remove a remote session mask
  TVMContext RemoveSessMask(TVMContext ctx) const {
    int dev_type = ctx.device_type;
    CHECK_EQ(dev_type / kRPCSessMask, sess_->table_index() + 1)
        << "Can not pass in local context or context with a different remote session";
    ctx.device_type = static_cast<DLDeviceType>(ctx.device_type % kRPCSessMask);
    return ctx;
  }

  // deleter of RPC remote array
  static void RemoteNDArrayDeleter(Object* obj) {
    auto* ptr = static_cast<NDArray::Container*>(obj);
    RemoteSpace* space = static_cast<RemoteSpace*>(ptr->dl_tensor.data);
    space->sess->FreeHandle(ptr->manager_ctx, kTVMNDArrayHandle);
    delete space;
    delete ptr;
  }

  // wrap return value as remote NDArray.
  NDArray WrapRemoteNDArray(DLTensor* tensor, void* nd_handle) const {
    NDArray::Container* data = new NDArray::Container();
    data->manager_ctx = nd_handle;
    data->SetDeleter(RemoteNDArrayDeleter);
    RemoteSpace* space = new RemoteSpace();
    space->sess = sess_;
    space->data = tensor->data;
    data->dl_tensor.data = space;
    NDArray ret(GetObjectPtr<Object>(data));
    // RAII now in effect
    data->shape_ = std::vector<int64_t>(tensor->shape, tensor->shape + tensor->ndim);
    data->dl_tensor.shape = dmlc::BeginPtr(data->shape_);
    data->dl_tensor.ndim = static_cast<int>(data->shape_.size());
    // setup dtype
    data->dl_tensor.dtype = tensor->dtype;
    // setup ctx, encode as remote session
    data->dl_tensor.ctx.device_id = tensor->ctx.device_id;
    data->dl_tensor.ctx.device_type = static_cast<DLDeviceType>(
        static_cast<int>(tensor->ctx.device_type) + kRPCSessMask * (sess_->table_index() + 1));
    // check strides.
    CHECK(tensor->strides == nullptr);
    // setup byteoffset
    data->dl_tensor.byte_offset = tensor->byte_offset;
    return ret;
  }
};

// RPC that represents a remote module session.
class RPCModuleNode final : public ModuleNode {
 public:
  RPCModuleNode(void* module_handle, std::shared_ptr<RPCSession> sess)
      : module_handle_(module_handle), sess_(sess) {}

  ~RPCModuleNode() {
    if (module_handle_ != nullptr) {
      try {
        sess_->FreeHandle(module_handle_, kTVMModuleHandle);
      } catch (const dmlc::Error& e) {
        // fault tolerance to remote close
      }
      module_handle_ = nullptr;
    }
  }

  const char* type_key() const final { return "rpc"; }

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final {
    if (module_handle_ == nullptr) {
      return WrapRemoteFunc(sess_->GetFunction(name));
    } else {
      InitRemoteFunc(&remote_mod_get_function_, "tvm.rpc.server.ModuleGetFunction");
      return remote_mod_get_function_(GetRef<Module>(this), name, false);
    }
  }

  std::string GetSource(const std::string& format) final {
    LOG(FATAL) << "GetSource for rpc Module is not supported";
    return "";
  }

  PackedFunc GetTimeEvaluator(const std::string& name, TVMContext ctx, int number, int repeat,
                              int min_repeat_ms) {
    InitRemoteFunc(&remote_get_time_evaluator_, "runtime.RPCTimeEvaluator");

    Optional<Module> opt_mod =
      (module_handle_ == nullptr)
        ? Optional<Module>(nullptr)
        : GetRef<Module>(this);

    return PackedFunc([this, opt_mod, name, ctx, number, repeat, min_repeat_ms] (TVMArgs args, TVMRetValue* rv) mutable {
      int num_flat_args = 7 + args.num_args;
      TVMValue values[num_flat_args];
      int type_codes[num_flat_args];
      TVMArgsSetter setter(values, type_codes);
      int offs = 0;
      setter(offs, this);
      offs++;
      setter(offs, opt_mod);
      offs++;
      setter(offs, name);
      offs++;
      setter(offs, ctx);
      offs++;
      setter(offs, number);
      offs++;
      setter(offs, repeat);
      offs++;
      setter(offs, min_repeat_ms);
      offs++;

      for (int i = 0; i < args.num_args; i++) {
        setter(offs + i, args[i]);
      }
      remote_get_time_evaluator_.CallPacked(TVMArgs(values, type_codes, num_flat_args), rv);
    });
  }

  Module LoadModule(std::string name) {
    InitRemoteFunc(&remote_load_module_, "tvm.rpc.server.load_module");
    return remote_load_module_(name);
  }

  void ImportModule(Module other) {
    InitRemoteFunc(&remote_import_module_, "tvm.rpc.server.ImportModule");
    remote_import_module_(GetRef<Module>(this), other);
  }

  const std::shared_ptr<RPCSession>& sess() { return sess_; }

  void* module_handle() const { return module_handle_; }

 private:
  template <typename FType>
  void InitRemoteFunc(FType* func, const std::string& name) {
    if (*func != nullptr) return;
    RPCSession::PackedFuncHandle handle = sess_->GetFunction(name);
    CHECK(handle != nullptr) << "Could not find remote function " << name;
    *func = WrapRemoteFunc(handle);
  }

  PackedFunc WrapRemoteFunc(RPCSession::PackedFuncHandle handle) {
    if (handle == nullptr) return PackedFunc();
    auto wf = std::make_shared<RPCWrappedFunc>(handle, sess_);
    return PackedFunc([wf](TVMArgs args, TVMRetValue* rv) { return wf->operator()(args, rv); });
  }

  // The module handle
  void* module_handle_{nullptr};
  // The local channel
  std::shared_ptr<RPCSession> sess_;
  // remote function to get time evaluator
  //   needs to be dynamically typed, because it's variadic
  PackedFunc remote_get_time_evaluator_;
  // remote function getter for modules.
  TypedPackedFunc<PackedFunc(Module, std::string, bool)> remote_mod_get_function_;
  // remote function getter for load module
  TypedPackedFunc<Module(std::string)> remote_load_module_;
  // remote function getter for load module
  TypedPackedFunc<void(Module, Module)> remote_import_module_;
};

void* RPCWrappedFunc::UnwrapRemoteValueToHandle(const TVMArgValue& arg) const {
  if (arg.type_code() == kTVMModuleHandle) {
    Module mod = arg;
    std::string tkey = mod->type_key();
    CHECK_EQ(tkey, "rpc") << "ValueError: Cannot pass a non-RPC module to remote";
    auto* rmod = static_cast<RPCModuleNode*>(mod.operator->());
    CHECK(rmod->sess() == sess_)
        << "ValueError: Cannot pass in module into a different remote session";
    return rmod->module_handle();
  } else {
    LOG(FATAL) << "ValueError: Cannot pass type " << runtime::ArgTypeCode2Str(arg.type_code())
               << " as an argument to the remote";
    return nullptr;
  }
}

void RPCWrappedFunc::WrapRemoteReturnToValue(TVMArgs args, TVMRetValue* rv) const {
  int tcode = args[0];

  if (tcode == kTVMNullptr) return;
  if (tcode == kTVMPackedFuncHandle) {
    CHECK_EQ(args.size(), 2);
    void* handle = args[1];
    auto wf = std::make_shared<RPCWrappedFunc>(handle, sess_);
    *rv = PackedFunc([wf](TVMArgs args, TVMRetValue* rv) { return wf->operator()(args, rv); });
  } else if (tcode == kTVMModuleHandle) {
    CHECK_EQ(args.size(), 2);
    void* handle = args[1];
    auto n = make_object<RPCModuleNode>(handle, sess_);
    *rv = Module(n);
  } else if (tcode == kTVMDLTensorHandle || tcode == kTVMNDArrayHandle) {
    CHECK_EQ(args.size(), 3);
    DLTensor* tensor = args[1];
    void* nd_handle = args[2];
    *rv = WrapRemoteNDArray(tensor, nd_handle);
  } else if (tcode == kTVMBytes) {
    // TODO stop handling bytes as strings internally?
    CHECK_EQ(args.size(), 2);
    const std::string& byte_arr = args[1];
    *rv = TVMByteArray {
      .data = byte_arr.data(),
      .size = byte_arr.size() - 1  // strip off null terminator
    };
  } else {
    CHECK_EQ(args.size(), 2);
    *rv = args[1];
  }
}

Module CreateRPCSessionModule(std::shared_ptr<RPCSession> sess) {
  auto n = make_object<RPCModuleNode>(nullptr, sess);
  RPCSession::InsertToSessionTable(sess);
  return Module(n);
}

std::shared_ptr<RPCSession> RPCModuleGetSession(Module mod) {
  std::string tkey = mod->type_key();
  CHECK_EQ(tkey, "rpc") << "ValueError: Cannot pass a non-RPC module to remote";
  auto* rmod = static_cast<RPCModuleNode*>(mod.operator->());
  return rmod->sess();
}

PackedFunc WrapTimeEvaluator(PackedFunc pf, TVMContext ctx, int number, int repeat,
                             int min_repeat_ms) {
  CHECK(pf != nullptr);

  auto ftimer = [pf, ctx, number, repeat, min_repeat_ms](TVMArgs args, TVMRetValue* rv) mutable {
    TVMRetValue temp;
    std::ostringstream os;
    // skip first time call, to activate lazy compilation components.
    pf.CallPacked(args, &temp);

    DeviceAPI::Get(ctx)->StreamSync(ctx, nullptr);

    for (int i = 0; i < repeat; ++i) {
      std::chrono::time_point<std::chrono::high_resolution_clock, std::chrono::nanoseconds> tbegin,
          tend;
      double duration_ms = 0.0;

      // do-while structure ensures we run even when `min_repeat_ms` isn't set (i.e., is 0).
      do {
        if (duration_ms > 0.0) {
          number = static_cast<int>(std::max((min_repeat_ms / (duration_ms / number) + 1),
                                             number * 1.618));  // 1.618 is chosen by random
        }

        tbegin = std::chrono::high_resolution_clock::now();
        // start timing
        for (int i = 0; i < number; ++i) {
          pf.CallPacked(args, &temp);
        }
        DeviceAPI::Get(ctx)->StreamSync(ctx, nullptr);
        tend = std::chrono::high_resolution_clock::now();

        duration_ms =
            std::chrono::duration_cast<std::chrono::duration<double>>(tend - tbegin).count() * 1000;
      } while (duration_ms < min_repeat_ms);

      double speed =
          std::chrono::duration_cast<std::chrono::duration<double>>(tend - tbegin).count() / number;
      os.write(reinterpret_cast<char*>(&speed), sizeof(speed));
    }
    std::string blob = os.str();
    TVMByteArray arr;
    arr.size = blob.length();
    arr.data = blob.data();
    // return the time.
    *rv = arr;
  };
  return PackedFunc(ftimer);
}

TVM_REGISTER_GLOBAL("runtime.RPCTimeEvaluator")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      Optional<Module> opt_mod = args[0];
      std::string name = args[1];
      int device_type = args[2];
      int device_id = args[3];
      int number = args[4];
      int repeat = args[5];
      int min_repeat_ms = args[6];

      TVMContext ctx;
      ctx.device_type = static_cast<DLDeviceType>(device_type);
      ctx.device_id = device_id;

      PackedFunc timer_func;
      if (opt_mod.defined()) {
        Module m = opt_mod.value();
        std::string tkey = m->type_key();
        if (tkey == "rpc") {
          timer_func = static_cast<RPCModuleNode*>(m.operator->())
              ->GetTimeEvaluator(name, ctx, number, repeat, min_repeat_ms);
        } else {
          timer_func = WrapTimeEvaluator(m.GetFunction(name, false), ctx, number, repeat, min_repeat_ms);
        }
      } else {
        auto* pf = runtime::Registry::Get(name);
        CHECK(pf != nullptr) << "Cannot find " << name << " in the global function";
        timer_func = WrapTimeEvaluator(*pf, ctx, number, repeat, min_repeat_ms);
      }
      timer_func.CallPacked(
        TVMArgs(&(args.values[7]), &(args.type_codes[7]), args.num_args - 7),
        rv);
    });

// server function registration.
TVM_REGISTER_GLOBAL("tvm.rpc.server.ImportModule").set_body_typed([](Module parent, Module child) {
  parent->Import(child);
});

TVM_REGISTER_GLOBAL("tvm.rpc.server.ModuleGetFunction")
    .set_body_typed([](Module parent, std::string name, bool query_imports) {
      return parent->GetFunction(name, query_imports);
    });

// functions to access an RPC module.
TVM_REGISTER_GLOBAL("rpc.LoadRemoteModule").set_body_typed([](Module sess, std::string name) {
  std::string tkey = sess->type_key();
  CHECK_EQ(tkey, "rpc");
  return static_cast<RPCModuleNode*>(sess.operator->())->LoadModule(name);
});

TVM_REGISTER_GLOBAL("rpc.ImportRemoteModule").set_body_typed([](Module parent, Module child) {
  std::string tkey = parent->type_key();
  CHECK_EQ(tkey, "rpc");
  static_cast<RPCModuleNode*>(parent.operator->())->ImportModule(child);
});

TVM_REGISTER_GLOBAL("rpc.SessTableIndex").set_body([](TVMArgs args, TVMRetValue* rv) {
  Module m = args[0];
  std::string tkey = m->type_key();
  CHECK_EQ(tkey, "rpc");
  *rv = static_cast<RPCModuleNode*>(m.operator->())->sess()->table_index();
});

}  // namespace runtime
}  // namespace tvm
