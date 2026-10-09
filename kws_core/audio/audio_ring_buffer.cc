// SPDX-License-Identifier: Apache-2.0
#include "audio/audio_ring_buffer.h"

namespace kws {

void AudioRingBuffer::Reset() {
  write_ = read_ = available_ = 0;
  overruns_ = 0;
  total_written_ = 0;
}

bool AudioRingBuffer::Push(const AudioSample* samples, size_t count) {
  if (samples == nullptr || count == 0) return true;

  bool ok = true;
  // A push larger than the ring can only keep its tail; that is already an
  // overrun by definition.
  if (count > Capacity()) {
    samples += (count - Capacity());
    count = Capacity();
    ok = false;
  }
  if (available_ + count > Capacity()) {
    ok = false;
  }

  for (size_t i = 0; i < count; ++i) {
    buffer_[write_] = samples[i];
    write_ = (write_ + 1) % Capacity();
  }
  total_written_ += count;

  if (!ok) {
    ++overruns_;
    // Oldest data was overwritten: the read cursor follows the write cursor.
    read_ = write_;
    available_ = Capacity();
  } else {
    available_ += count;
  }
  return ok;
}

bool AudioRingBuffer::Read(size_t offset_from_oldest, AudioSample* out, size_t count) const {
  if (out == nullptr) return false;
  if (offset_from_oldest + count > available_) return false;
  size_t idx = (read_ + offset_from_oldest) % Capacity();
  for (size_t i = 0; i < count; ++i) {
    out[i] = buffer_[idx];
    idx = (idx + 1) % Capacity();
  }
  return true;
}

bool AudioRingBuffer::PeekNewest(AudioSample* out, size_t count) const {
  if (out == nullptr || count > Capacity()) return false;
  if (total_written_ < count) return false;
  size_t idx = (write_ + Capacity() - count) % Capacity();
  for (size_t i = 0; i < count; ++i) {
    out[i] = buffer_[idx];
    idx = (idx + 1) % Capacity();
  }
  return true;
}

void AudioRingBuffer::Advance(size_t count) {
  if (count > available_) count = available_;
  read_ = (read_ + count) % Capacity();
  available_ -= count;
}

}  // namespace kws
