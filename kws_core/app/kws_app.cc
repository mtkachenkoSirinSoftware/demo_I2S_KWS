// SPDX-License-Identifier: Apache-2.0
#include "app/kws_app.h"

#include "audio/audio_ring_buffer.h"
#include "dsp/feature_ring_buffer.h"
#include "dsp/streaming_frontend.h"
#include "model/model_runner.h"
#include "postprocessing/recognizer.h"
#include "profiling/kws_profile.h"

namespace kws {
namespace {
// All steady-state storage is static. Sizes live in kws_config.h.
AudioRingBuffer g_pcm;
StreamingFrontend g_frontend;
FeatureRingBuffer g_features;
ModelRunner g_model;
Recognizer g_recognizer;

// Scratch for one analysis frame and one MFCC window. Static, not stack, so the
// worst-case stack depth does not depend on the frontend.
AudioSample g_frame[KWS_FRAME_LENGTH];
float g_mfcc[KWS_NUM_FRAMES * KWS_NUM_MFCC];
AudioSample g_window[KWS_CLIP_SAMPLES];

// Samples consumed by the frontend so far, i.e. the end of the last analysed
// frame relative to the ring's read cursor.
size_t g_analysed = 0;
}  // namespace

bool KwsApp::ResetSession() {
  g_pcm.Reset();
  g_features.Reset();
  if (!g_frontend.Init()) return false;
  g_recognizer.Init();
  g_analysed = 0;
  telemetry_ = Telemetry();
  frames_since_inference_ = 0;
  last_label_ = -1;
  last_score_ = 0.0f;
  last_event_ = KeywordEvent();
  return true;
}

bool KwsApp::Init() {
  if (!ResetSession()) return false;
  return g_model.Init();
}

uint32_t KwsApp::OnAudioBlock(const AudioSample* samples, uint32_t count, uint32_t now_ms) {
  (void)now_ms;
  if (samples == nullptr || count == 0) return 0;

  if (!g_pcm.Push(samples, count)) {
    ++telemetry_.audio_overruns;
  }
  ++telemetry_.audio_blocks_received;

  // Emit every frame that has become complete. Only the new frames are
  // analysed; nothing already in the feature ring is recomputed.
  uint32_t produced = 0;
  while (g_analysed + KWS_FRAME_LENGTH <= g_pcm.Available()) {
    if (!g_pcm.Read(g_analysed, g_frame, KWS_FRAME_LENGTH)) break;
    {
      CycleScope t(&telemetry_.frontend_cycles_last);
      g_frontend.PushFrame(g_frame, &g_features);
    }
    g_analysed += KWS_FRAME_STEP;
    ++produced;
    ++telemetry_.frontend_frames_generated;
    ++frames_since_inference_;
  }

  // Release PCM the frontend can no longer need: everything older than the
  // current inference window plus one frame of overlap.
  const size_t keep = KWS_CLIP_SAMPLES + KWS_FRAME_LENGTH;
  if (g_pcm.Available() > keep) {
    const size_t drop = g_pcm.Available() - keep;
    g_pcm.Advance(drop);
    g_analysed = (g_analysed > drop) ? (g_analysed - drop) : 0;
  }

  ++telemetry_.audio_blocks_processed;
  return produced;
}

bool KwsApp::InferenceDue() const {
  return g_features.WindowReady() && frames_since_inference_ >= KWS_INFERENCE_STRIDE_FRAMES;
}

bool KwsApp::WindowReady() const { return g_features.WindowReady(); }

bool KwsApp::FlushInference(uint32_t now_ms) {
  if (!WindowReady()) return false;
  // Last RunInference already consumed this window (stride just elapsed).
  if (frames_since_inference_ == 0) return false;
  return RunInference(now_ms);
}

bool KwsApp::RunInference(uint32_t now_ms) {
  if (!g_features.WindowReady()) return false;
  frames_since_inference_ = 0;

  // The peak normaliser is defined over the same 1 s the model sees.
  float peak = 0.0f;
  {
    CycleScope t(&telemetry_.mfcc_cycles_last);
    if (g_pcm.PeekNewest(g_window, KWS_CLIP_SAMPLES)) {
      peak = ClipPeak(g_window, KWS_CLIP_SAMPLES);
    }
    if (!g_frontend.ComputeMfccWindow(g_features, peak, g_mfcc)) return false;
  }
  {
    CycleScope t(&telemetry_.quantize_cycles_last);
    if (!g_model.SetInput(g_mfcc)) return false;
  }
  {
    // Times the runtime's Invoke() as the application issues it. Under
    // heliaRT that is one MicroInterpreter::Invoke() over an already-allocated
    // arena; under nsx-executorch it also includes program load, because
    // run_once() has no persistent-method form. Comparing the two without
    // saying so would be comparing different quantities.
    CycleScope t(&telemetry_.inference_cycles_last);
    if (!g_model.Invoke()) return false;
  }

  const Prediction p = g_model.GetOutput();
  last_label_ = p.label;
  last_score_ = p.score;
  ++telemetry_.inferences_run;

  const KeywordEvent ev = g_recognizer.Update(p.scores, now_ms);
  last_event_ = ev;
  if (ev.fired) {
    ++telemetry_.events_fired;
    return true;
  }
  return false;
}

}  // namespace kws
