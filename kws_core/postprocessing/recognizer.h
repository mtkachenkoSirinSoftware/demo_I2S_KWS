// SPDX-License-Identifier: Apache-2.0
// Temporal post-processing: a spoken keyword must produce one event,
// not one per overlapping inference window.
#ifndef RECOGNIZER_H_
#define RECOGNIZER_H_

#include <stdint.h>

#include "config/kws_config.h"

namespace kws {

struct KeywordEvent {
  bool fired = false;
  int label = -1;
  float score = 0.0f;
  uint32_t timestamp_ms = 0;
};

// Pipeline: raw scores -> moving average -> threshold -> consistency -> debounce.
class Recognizer {
 public:
  void Init();
  void Reset();

  // Feeds one inference result. `timestamp_ms` is the monotonic time of the
  // window's newest sample.
  KeywordEvent Update(const float* scores, uint32_t timestamp_ms);

  void set_threshold(float t) { threshold_ = t; }
  void set_suppression_ms(uint32_t ms) { suppression_ms_ = ms; }
  void set_min_consistent(uint32_t n) { min_consistent_ = n; }

  uint32_t events_fired() const { return events_fired_; }

 private:
  float history_[KWS_SMOOTHING_WINDOW][KWS_NUM_CLASSES] = {};
  uint32_t history_count_ = 0;
  uint32_t history_head_ = 0;

  int consistent_label_ = -1;
  uint32_t consistent_count_ = 0;

  bool have_fired_ = false;
  uint32_t last_event_ms_ = 0;

  float threshold_ = KWS_DETECTION_THRESHOLD;
  uint32_t suppression_ms_ = KWS_SUPPRESSION_MS;
  uint32_t min_consistent_ = KWS_MIN_CONSISTENT;
  uint32_t events_fired_ = 0;
};

}  // namespace kws
#endif  // RECOGNIZER_H_
