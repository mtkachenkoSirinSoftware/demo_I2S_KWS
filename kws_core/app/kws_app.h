// SPDX-License-Identifier: Apache-2.0
#ifndef KWS_APP_H_
#define KWS_APP_H_

#include <stdint.h>

#include "config/kws_config.h"
#include "postprocessing/recognizer.h"

namespace kws {

struct Telemetry {
  uint32_t audio_blocks_received = 0;
  uint32_t audio_blocks_processed = 0;
  uint32_t audio_overruns = 0;
  uint32_t uart_errors = 0;
  uint32_t frontend_frames_generated = 0;
  uint32_t inferences_run = 0;
  uint32_t events_fired = 0;
  // Cycle counts, not microseconds: cycles are what the target actually
  // reports (profiling/kws_profile.h), and converting to time needs a clock
  // the platform has to hand us rather than one this struct assumes. On a host
  // build every one of these stays 0 by design -- see profiling/kws_profile.h.
  uint32_t frontend_cycles_last = 0;   // one hop: window + FFT + mel
  uint32_t mfcc_cycles_last = 0;       // per inference: peak + log + DCT
  uint32_t quantize_cycles_last = 0;   // per inference: float -> int8
  uint32_t inference_cycles_last = 0;  // per inference: the runtime's Invoke()
};

// The whole pipeline, source-agnostic. Nothing here knows whether the PCM came
// from a microphone or a UART.
class KwsApp {
 public:
  bool Init();

  // UART T_RESET / new clip: rings, frontend tables, recognizer. Does **not**
  // reconstruct the helia interpreter (second AllocateTensors / placement-new
  // hangs on target — silicon: stats freeze at 0 after host RESET).
  bool ResetSession();

  // Feeds one acquisition block. Returns the number of feature frames produced.
  uint32_t OnAudioBlock(const AudioSample* samples, uint32_t count, uint32_t now_ms);

  // True when enough new frames have accumulated for the next sliding window.
  bool InferenceDue() const;

  // Runs one inference + post-processing. Returns true if a keyword fired.
  bool RunInference(uint32_t now_ms);

  // Feature ring holds a full 1 s window. Used to flush the last window on
  // UART END_STREAM even when the inference stride has not elapsed.
  bool WindowReady() const;

  // Like RunInference but ignores the hop stride. No-op if !WindowReady().
  bool FlushInference(uint32_t now_ms);

  const Telemetry& telemetry() const { return telemetry_; }
  int last_label() const { return last_label_; }
  float last_score() const { return last_score_; }
  const KeywordEvent& last_event() const { return last_event_; }

 private:
  Telemetry telemetry_;
  uint32_t frames_since_inference_ = 0;
  int last_label_ = -1;
  float last_score_ = 0.0f;
  KeywordEvent last_event_{};
};

}  // namespace kws
#endif  // KWS_APP_H_
