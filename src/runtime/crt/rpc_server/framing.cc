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
 * \file framing.cc
 * \brief Framing for RPC.
 */

#include "framing.h"

//#include <type_traits>
#include <string.h>
#include <tvm/runtime/crt/logging.h>
#include "crt_config.h"

// For debugging purposes, Framer logs can be enabled, but this should only be done when
// running from the host.
#ifdef TVM_CRT_FRAMER_ENABLE_LOGS
#include <cstdio>
#define TVM_FRAMER_DEBUG_LOG(msg, ...)  fprintf(stderr, "utvm framer: " msg " \n", ##__VA_ARGS__)
#else
#define TVM_FRAMER_DEBUG_LOG(msg, ...)
#endif

namespace tvm {
namespace runtime {

//template <typename E>
//static constexpr typename std::underlying_type<E>::type to_integral(E e) {
//  return static_cast<typename std::underlying_type<E>::type>(e);
//}

template <typename E>
static constexpr uint8_t to_integral(E e) {
  return static_cast<uint8_t>(e);
}

int Unframer::Write(const uint8_t* data, size_t data_size_bytes, size_t* bytes_consumed) {
  int return_code = 0;
  input_ = data;
  input_size_bytes_ = data_size_bytes;

  while (return_code == 0 && input_size_bytes_ > 0) {
    TVM_FRAMER_DEBUG_LOG("state: %02x size %02lx", to_integral(state_), input_size_bytes_);
    switch (state_) {
    case State::kFindPacketStart:
      return_code = FindPacketStart();
      break;
    case State::kFindPacketLength:
      return_code = FindPacketLength();
      break;
    case State::kFindPacketCrc:
      return_code = FindPacketCrc();
      break;
    case State::kFindCrcEnd:
      return_code = FindCrcEnd();
      break;
    default:
      return_code = -1;
      break;
    }
  }

  *bytes_consumed = data_size_bytes - input_size_bytes_;
  input_ = nullptr;
  input_size_bytes_ = 0;

  if (return_code == -1) {
    state_ = State::kFindPacketStart;
    ClearBuffer();
  }

  return return_code;
}

int Unframer::FindPacketStart() {
  size_t i;
  for (i = 0; i < input_size_bytes_; i++) {
    if (input_[i] == to_integral(Escape::kEscapeStart)) {
      saw_escape_start_ = !saw_escape_start_;
    } else if (input_[i] == to_integral(Escape::kPacketStart) && saw_escape_start_) {
      uint8_t packet_start_sequence[2]{to_integral(Escape::kEscapeStart), to_integral(Escape::kPacketStart)};
      crc_ = crc16_compute(packet_start_sequence, sizeof(packet_start_sequence), nullptr);
      saw_escape_start_ = false;
      state_ = State::kFindPacketLength;
      i++;
      break;
    } else {
      saw_escape_start_ = false;
    }
  }

  input_ += i;
  input_size_bytes_ -= i;
  return 0;
}

int Unframer::ConsumeInput(uint8_t* buffer, size_t buffer_size_bytes, size_t* bytes_filled, bool update_crc) {
  CHECK(*bytes_filled < buffer_size_bytes);
  int to_return = 0;
  size_t i;
  for (i = 0; i < input_size_bytes_; i++) {
    uint8_t c = input_[i];
    if (saw_escape_start_) {
      saw_escape_start_ = false;
      if (c == to_integral(Escape::kPacketStart)) {
        // When the start packet sequence is seen, abort unframing the current packet. Since the
        // escape byte has already been parsed, update the CRC include only the escape byte. This
        // readies the unframer to consume the kPacketStart byte on the next Write() call.
        uint8_t escape_start = to_integral(Escape::kEscapeStart);
        crc_ = crc16_compute(&escape_start, 1, NULL);
        to_return = -1;
        saw_escape_start_ = true;

        break;
      } else if (c == to_integral(Escape::kEscapeNop)) {
        continue;
      } else if (c == to_integral(Escape::kEscapeStart)) {
        // do nothing (allow character to be printed)
      } else {
        // Invalid escape sequence.
        TVMLogf("invalid escape: %02x\n", c);
        to_return = -1;
        i++;
        break;
      }
    } else if (c == to_integral(Escape::kEscapeStart)) {
      saw_escape_start_ = true;
      continue;
    } else {
      saw_escape_start_ = false;
    }

    buffer[*bytes_filled] = c;
    (*bytes_filled)++;
    if (*bytes_filled == buffer_size_bytes) {
      i++;
      break;
    }
  }

  if (update_crc) {
    crc_ = crc16_compute(input_, i, &crc_);
  }

  input_ += i;
  input_size_bytes_ -= i;
  return to_return;
}

int Unframer::AddToBuffer(size_t buffer_full_bytes, bool update_crc) {
  CHECK(!is_buffer_full(buffer_full_bytes));
  return ConsumeInput(
    buffer_, buffer_full_bytes, &num_buffer_bytes_valid_, update_crc);
}

void Unframer::ClearBuffer() {
  num_buffer_bytes_valid_ = 0;
}

int Unframer::FindPacketLength() {
  int to_return = AddToBuffer(PacketFieldSizeBytes::kPayloadLength, true);
  if (to_return != 0) {
    return to_return;
  }

  if (!is_buffer_full(PacketFieldSizeBytes::kPayloadLength)) {
    return 0;
  }

  // TODO endian
  num_payload_bytes_remaining_ = *((uint32_t*) buffer_);
  TVM_FRAMER_DEBUG_LOG("packet length: %08zu", num_payload_bytes_remaining_);
  ClearBuffer();
  state_ = State::kFindPacketCrc;
  return 0;
};


int Unframer::FindPacketCrc() {
//  CHECK(num_buffer_bytes_valid_ == 0);
  TVM_FRAMER_DEBUG_LOG("find packet crc: %02zu", num_payload_bytes_remaining_);
  while (num_payload_bytes_remaining_ > 0) {
    size_t num_bytes_to_buffer = num_payload_bytes_remaining_;
    if (num_bytes_to_buffer > sizeof(buffer_)) {
      num_bytes_to_buffer = sizeof(buffer_);
    }

    size_t prev_num_buffer_bytes_valid = num_buffer_bytes_valid_;
    {
      int to_return = AddToBuffer(num_bytes_to_buffer, true);
      if (to_return != 0) {
        return to_return;
      }
    }

    if (prev_num_buffer_bytes_valid == num_buffer_bytes_valid_) {
      // Return if no bytes were consumed from the input.
      return 0;
    }

    {
      int to_return = stream_->WriteAll(buffer_, num_buffer_bytes_valid_);
      if (to_return != 0) {
        return to_return;
      }
    }

    num_payload_bytes_remaining_ -= num_buffer_bytes_valid_;
    ClearBuffer();
  }

  if (num_payload_bytes_remaining_ == 0) {
    state_ = State::kFindCrcEnd;
  }

  return 0;
}

int Unframer::FindCrcEnd() {
  int to_return = AddToBuffer(PacketFieldSizeBytes::kCrc, false);
  if (to_return != 0) {
    return to_return;
  }

  if (!is_buffer_full(PacketFieldSizeBytes::kCrc)) {
    return 0;
  }

  // TODO endian
  stream_->PacketDone(crc_ == *((uint16_t*) buffer_));
  ClearBuffer();
  state_ = State::kFindPacketStart;
  return 1;
}


int Framer::Write(const uint8_t* payload, size_t payload_size_bytes) {
  int to_return;
  to_return = StartPacket(payload_size_bytes);
  if (to_return != 0) {
    return to_return;
  }

  to_return = WritePayloadChunk(payload, payload_size_bytes);
  if (to_return != 0) {
    return to_return;
  }

  to_return = FinishPacket();
  return to_return;
}

int Framer::StartPacket(size_t payload_size_bytes) {
  uint8_t packet_header[sizeof(uint32_t)];
  size_t ptr = 0;
  if (state_ == State::kReset) {
    packet_header[ptr] = to_integral(Escape::kEscapeNop);
    ptr++;
    int to_return = WriteAndCrc(packet_header, ptr, false  /* escape */, false /* update_crc */);
    if (to_return != 0) {
      return to_return;
    }

    ptr = 0;
  }

  packet_header[ptr] = to_integral(Escape::kEscapeStart);
  ptr++;
  packet_header[ptr] = to_integral(Escape::kPacketStart);
  ptr++;

  crc_ = 0xffff;
  int to_return = WriteAndCrc(
    packet_header, ptr, false  /* escape */, true  /* update_crc */);
  if (to_return != 0) {
    return to_return;
  }

  uint32_t payload_size_wire = payload_size_bytes;
  to_return = WriteAndCrc(
    (uint8_t*) &payload_size_wire, sizeof(payload_size_wire), true  /* escape */, true  /* update_crc */);
  if (to_return == 0) {
    state_ = State::kTransmitPacketPayload;
    num_payload_bytes_remaining_ = payload_size_bytes;
  }

  return to_return;
}

int Framer::WriteAndCrc(const uint8_t* data, size_t data_size_bytes, bool escape, bool update_crc) {
  while (data_size_bytes > 0) {
    uint8_t buffer[kMaxStackBufferSizeBytes];
    size_t buffer_ptr = 0;
    size_t i;
    for (i = 0; i < data_size_bytes && buffer_ptr != kMaxStackBufferSizeBytes; i++) {
      uint8_t c = data[i];
      if (!escape || c != to_integral(Escape::kEscapeStart)) {
        buffer[buffer_ptr] = c;
        buffer_ptr++;
        continue;
      }

      if (buffer_ptr == kMaxStackBufferSizeBytes - 1) {
        break;
      }

      buffer[buffer_ptr] = to_integral(Escape::kEscapeStart);
      buffer_ptr++;

      buffer[buffer_ptr] = to_integral(Escape::kEscapeStart);
      buffer_ptr++;
    }

    ssize_t to_return = stream_->WriteAll(buffer, buffer_ptr);
    if (to_return < 0) {
      return to_return;
    }

    if (update_crc) {
      crc_ = crc16_compute(buffer, buffer_ptr, &crc_);
    }

    data_size_bytes -= i;
    data += i;
  }

  return 0;
}

int Framer::WritePayloadChunk(const uint8_t* payload_chunk, size_t payload_chunk_size_bytes) {
  if (state_ != State::kTransmitPacketPayload ||
      payload_chunk_size_bytes > num_payload_bytes_remaining_) {
    return -1;
  }

  TVM_FRAMER_DEBUG_LOG("write payload chunk: %" PRIuMAX " bytes\n", payload_chunk_size_bytes);
  int to_return = WriteAndCrc(
    payload_chunk, payload_chunk_size_bytes, true  /* escape */, true  /* update_crc */);
  if (to_return != 0) {
    state_ = State::kReset;
    return to_return;
  }

  num_payload_bytes_remaining_ -= payload_chunk_size_bytes;
  return 0;
}

int Framer::FinishPacket() {
  if (state_ != State::kTransmitPacketPayload || num_payload_bytes_remaining_ != 0) {
    return -1;
  }

  int to_return = WriteAndCrc(
    reinterpret_cast<uint8_t*>(&crc_), sizeof(crc_), true  /* escape */, false  /* update_crc */);
  if (to_return != 0) {
    TVM_FRAMER_DEBUG_LOG("write and crc returned: %02x", to_return);
    state_ = State::kReset;
  } else {
    state_ = State::kIdle;
  }
  return to_return;
}

}  // namespace runtime
}  // namespace tvm
