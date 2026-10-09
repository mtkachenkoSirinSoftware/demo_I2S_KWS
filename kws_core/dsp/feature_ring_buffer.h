// SPDX-License-Identifier: Apache-2.0
// Feature ring. Circular indexing only — the complete history is never memmove'd.
//
// What is stored is one *linear mel energy* vector per frame, not the finished
// MFCC. The training frontend divides the whole 1 s clip by max(x) before the log, so the
// log/DCT stage depends on a value that is only known once the window is
// complete. Mel energies are the last representation that is still causal, and
// they are linear in the input, so the normaliser can be applied later without
// approximation.
#ifndef FEATURE_RING_BUFFER_H_
#define FEATURE_RING_BUFFER_H_

#include <stddef.h>
#include <stdint.h>

#include "config/kws_config.h"

namespace kws {

class FeatureRingBuffer {
 public:
  static constexpr size_t kFrameWidth = KWS_NUM_MEL;
  static constexpr size_t kCapacityFrames = KWS_FEATURE_RING_FRAMES;

  void Reset();

  // Appends one frame (kFrameWidth floats). Oldest frame is dropped when full.
  void PushFrame(const float* frame);

  size_t count() const { return count_; }
  uint64_t total_frames() const { return total_frames_; }
  bool WindowReady() const { return count_ >= KWS_NUM_FRAMES; }

  // Read-only pointer to the frame `frames_back` positions before the newest
  // (0 == newest). Returns nullptr when out of range. No copy: the caller reads
  // straight out of the ring storage.
  const float* FrameFromNewest(size_t frames_back) const;

  // Frame `i` of the most recent KWS_NUM_FRAMES window, oldest first.
  const float* WindowFrame(size_t i) const;

 private:
  float frames_[kCapacityFrames][kFrameWidth] = {};
  size_t head_ = 0;   // next slot to write
  size_t count_ = 0;  // valid frames held
  uint64_t total_frames_ = 0;
};

}  // namespace kws
#endif  // FEATURE_RING_BUFFER_H_
