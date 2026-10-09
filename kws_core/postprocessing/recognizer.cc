// SPDX-License-Identifier: Apache-2.0
#include "postprocessing/recognizer.h"

#include "model/model_runner.h"

namespace kws {
namespace {
// _silence_ and _unknown_ are never reported as keyword events.
inline bool IsKeyword(int label) {
  return label >= 0 && label < KWS_NUM_CLASSES && label != 10 && label != 11;
}
}  // namespace

void Recognizer::Init() { Reset(); }

void Recognizer::Reset() {
  history_count_ = 0;
  history_head_ = 0;
  consistent_label_ = -1;
  consistent_count_ = 0;
  have_fired_ = false;
  last_event_ms_ = 0;
  events_fired_ = 0;
}

KeywordEvent Recognizer::Update(const float* scores, uint32_t timestamp_ms) {
  KeywordEvent ev;
  if (scores == nullptr) return ev;

  // 1. moving average over the last KWS_SMOOTHING_WINDOW inferences
  for (int c = 0; c < KWS_NUM_CLASSES; ++c) history_[history_head_][c] = scores[c];
  history_head_ = (history_head_ + 1) % KWS_SMOOTHING_WINDOW;
  if (history_count_ < KWS_SMOOTHING_WINDOW) ++history_count_;

  float smoothed[KWS_NUM_CLASSES] = {};
  for (uint32_t h = 0; h < history_count_; ++h) {
    for (int c = 0; c < KWS_NUM_CLASSES; ++c) smoothed[c] += history_[h][c];
  }
  const float inv = 1.0f / static_cast<float>(history_count_);
  for (int c = 0; c < KWS_NUM_CLASSES; ++c) smoothed[c] *= inv;

  int best = 0;
  for (int c = 1; c < KWS_NUM_CLASSES; ++c) {
    if (smoothed[c] > smoothed[best]) best = c;
  }

  // 2. threshold + 3. consistency
  if (smoothed[best] < threshold_ || !IsKeyword(best)) {
    consistent_label_ = -1;
    consistent_count_ = 0;
    return ev;
  }
  if (best == consistent_label_) {
    ++consistent_count_;
  } else {
    consistent_label_ = best;
    consistent_count_ = 1;
  }
  if (consistent_count_ < min_consistent_) return ev;

  // 4. suppression / debounce
  if (have_fired_ && (timestamp_ms - last_event_ms_) < suppression_ms_) return ev;

  ev.fired = true;
  ev.label = best;
  ev.score = smoothed[best];
  ev.timestamp_ms = timestamp_ms;
  have_fired_ = true;
  last_event_ms_ = timestamp_ms;
  ++events_fired_;
  return ev;
}

}  // namespace kws
