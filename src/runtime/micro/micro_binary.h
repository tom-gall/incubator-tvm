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
 * \file micro_binary.h
 * \brief Defines the MicroBinary class.
 */
#ifndef TVM_RUNTIME_MICRO_MICRO_BINARY_H_
#define TVM_RUNTIME_MICRO_MICRO_BINARY_H_

#include <tvm/runtime/object.h>

namespace tvm {
namespace runtime {

class MicroBinary : public ObjectRef {
 public:
  MicroBinary() {}
  explicit MicroBinary(ObjectRef<Object> n) : ObjectRef(n) {}

  /*!
   * \brief Get member function for use by frontend.
   *
   * \param name The name of the function.
   * \return The result function.
   *  This function will return PackedFunc(nullptr) if function do not exist.
   */
  inline PackedFunc GetFunction(const std::string& name);

  using ContainerType = MicroBinaryNode;
  friend class MicroBinaryNode;
};


class MicroBinaryNode : public Object {
 public:
  /*! \brief The binary file, encoded as ELF. */
  std::string elf_data;

  /*! \brief The names of callable tasks in this binary. */
  std::vector<string> task_names;
};

}  // namespace runtime
}  // namespace tvm

#endif
