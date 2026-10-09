// SPDX-License-Identifier: Apache-2.0
#include "dsp/feature_ring_buffer.h"

namespace kws {

void FeatureRingBuffer::Reset() {
  head_ = 0;
  count_ = 0;
  total_frames_ = 0;
}

void FeatureRingBuffer::PushFrame(const float* frame) {
  if (frame == nullptr) return;
  for (size_t i = 0; i < kFrameWidth; ++i) frames_[head_][i] = frame[i];
  head_ = (head_ + 1) % kCapacityFrames;
  if (count_ < kCapacityFrames) ++count_;
  ++total_frames_;
}

const float* FeatureRingBuffer::FrameFromNewest(size_t frames_back) const {
  if (frames_back >= count_) return nullptr;
  size_t idx = (head_ + kCapacityFrames - 1 - frames_back) % kCapacityFrames;
  return frames_[idx];
}

const float* FeatureRingBuffer::WindowFrame(size_t i) const {
  if (i >= KWS_NUM_FRAMES || !WindowReady()) return nullptr;
  // i == 0 is the oldest frame of the newest KWS_NUM_FRAMES-long window.
  return FrameFromNewest(KWS_NUM_FRAMES - 1 - i);
}

}  // namespace kws
