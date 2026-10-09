// SPDX-License-Identifier: Apache-2.0
// End-to-end behaviour of the source-agnostic pipeline.
#include "app/kws_app.h"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "config/kws_config.h"

using kws::KwsApp;

namespace {

std::vector<AudioSample> Tone(int n, float hz, float amp = 8000.0f) {
  std::vector<AudioSample> x(n);
  for (int i = 0; i < n; ++i) {
    x[i] = static_cast<AudioSample>(
        amp * sinf(2.0f * 3.14159265f * hz * static_cast<float>(i) / KWS_SAMPLE_RATE_HZ));
  }
  return x;
}

// Streams `seconds` of audio one DMA block at a time, exactly as main() does.
struct RunResult {
  uint32_t frames = 0;
  uint32_t inferences = 0;
};

RunResult Stream(KwsApp* app, const std::vector<AudioSample>& audio) {
  RunResult r;
  uint32_t t_ms = 0;
  for (size_t off = 0; off + KWS_AUDIO_BLOCK_SAMPLES <= audio.size();
       off += KWS_AUDIO_BLOCK_SAMPLES) {
    r.frames += app->OnAudioBlock(audio.data() + off, KWS_AUDIO_BLOCK_SAMPLES, t_ms);
    if (app->InferenceDue()) {
      app->RunInference(t_ms);
      ++r.inferences;
    }
    t_ms += KWS_AUDIO_BLOCK_MS;
  }
  return r;
}

}  // namespace

TEST(KwsApp, InitSucceeds) {
  KwsApp app;
  EXPECT_TRUE(app.Init());
}

TEST(KwsApp, ResetSessionDoesNotRequireReinit) {
  KwsApp app;
  ASSERT_TRUE(app.Init());
  auto audio = Tone(KWS_AUDIO_BLOCK_SAMPLES, 440.0f);
  EXPECT_EQ(app.OnAudioBlock(audio.data(), KWS_AUDIO_BLOCK_SAMPLES, 0), 0u);
  ASSERT_TRUE(app.ResetSession());
  EXPECT_EQ(app.telemetry().audio_blocks_received, 0u);
  EXPECT_EQ(app.OnAudioBlock(audio.data(), KWS_AUDIO_BLOCK_SAMPLES, 0), 0u);
  EXPECT_EQ(app.telemetry().audio_blocks_received, 1u);
}

TEST(KwsApp, ProducesOneFeatureFramePerBlockOncePrimed) {
  KwsApp app;
  ASSERT_TRUE(app.Init());
  auto audio = Tone(KWS_SAMPLE_RATE_HZ * 3, 440.0f);
  const RunResult r = Stream(&app, audio);

  // Block size == hop, so after the initial fill each block yields one frame.
  const uint32_t blocks = static_cast<uint32_t>(audio.size() / KWS_AUDIO_BLOCK_SAMPLES);
  EXPECT_GE(r.frames, blocks - 2);
  EXPECT_LE(r.frames, blocks);
  EXPECT_EQ(app.telemetry().frontend_frames_generated, r.frames);
}

TEST(KwsApp, NoInferenceBeforeAFullWindowExists) {
  KwsApp app;
  ASSERT_TRUE(app.Init());
  // Less than 1 s of audio: the feature ring can never hold a full window.
  auto audio = Tone(KWS_SAMPLE_RATE_HZ / 2, 440.0f);
  const RunResult r = Stream(&app, audio);
  EXPECT_EQ(r.inferences, 0u);
  EXPECT_EQ(app.telemetry().inferences_run, 0u);
}

// Inference cadence is decoupled from and slower than the
// frontend cadence, and acquisition is never paused for it.
TEST(KwsApp, InferenceRunsAtTheConfiguredStride) {
  KwsApp app;
  ASSERT_TRUE(app.Init());
  auto audio = Tone(KWS_SAMPLE_RATE_HZ * 5, 440.0f);
  const RunResult r = Stream(&app, audio);

  ASSERT_GT(r.inferences, 0u);
  // One inference per KWS_INFERENCE_STRIDE_FRAMES frames, after the first
  // window has been assembled.
  const double expected =
      static_cast<double>(r.frames - KWS_NUM_FRAMES) / KWS_INFERENCE_STRIDE_FRAMES;
  EXPECT_NEAR(static_cast<double>(r.inferences), expected, 2.0);
  EXPECT_LT(r.inferences, r.frames);
}

TEST(KwsApp, SustainedStreamingCausesNoOverruns) {
  KwsApp app;
  ASSERT_TRUE(app.Init());
  // 30 s of continuous audio through the ring + frontend.
  auto audio = Tone(KWS_SAMPLE_RATE_HZ * 30, 700.0f);
  Stream(&app, audio);
  EXPECT_EQ(app.telemetry().audio_overruns, 0u)
      << "PCM ring cannot keep up with block-at-a-time streaming";
}

TEST(KwsApp, TelemetryCountersAreConsistent) {
  KwsApp app;
  ASSERT_TRUE(app.Init());
  auto audio = Tone(KWS_SAMPLE_RATE_HZ * 4, 1000.0f);
  Stream(&app, audio);
  const auto& t = app.telemetry();
  EXPECT_EQ(t.audio_blocks_received, t.audio_blocks_processed);
  EXPECT_GT(t.frontend_frames_generated, 0u);
  EXPECT_GT(t.inferences_run, 0u);
  EXPECT_LE(t.inferences_run, t.frontend_frames_generated);
}

TEST(KwsApp, ProducesAPredictionAfterTheFirstWindow) {
  KwsApp app;
  ASSERT_TRUE(app.Init());
  auto audio = Tone(KWS_SAMPLE_RATE_HZ * 2, 440.0f);
  Stream(&app, audio);
  EXPECT_GE(app.last_label(), 0);
  EXPECT_LT(app.last_label(), KWS_NUM_CLASSES);
  EXPECT_GE(app.last_score(), 0.0f);
  EXPECT_LE(app.last_score(), 1.0f);
}

TEST(KwsApp, HandlesNullAndEmptyBlocks) {
  KwsApp app;
  ASSERT_TRUE(app.Init());
  EXPECT_EQ(app.OnAudioBlock(nullptr, KWS_AUDIO_BLOCK_SAMPLES, 0), 0u);
  AudioSample s = 0;
  EXPECT_EQ(app.OnAudioBlock(&s, 0, 0), 0u);
}

// Nothing may allocate after Init(). A crude but effective proxy is
// that a long run does not change the process's steady-state behaviour; the
// stronger guarantee comes from the code containing no new/malloc at all.
TEST(KwsApp, FlushInferenceRunsWhenStrideHasNotElapsed) {
  KwsApp app;
  ASSERT_TRUE(app.Init());
  auto audio = Tone(KWS_SAMPLE_RATE_HZ, 440.0f);
  uint32_t t_ms = 0;
  uint32_t inferences = 0;
  for (size_t off = 0; off + KWS_AUDIO_BLOCK_SAMPLES <= audio.size();
       off += KWS_AUDIO_BLOCK_SAMPLES) {
    app.OnAudioBlock(audio.data() + off, KWS_AUDIO_BLOCK_SAMPLES, t_ms);
    if (app.InferenceDue()) {
      app.RunInference(t_ms);
      ++inferences;
    }
    t_ms += KWS_AUDIO_BLOCK_MS;
  }
  ASSERT_TRUE(app.WindowReady());
  const uint32_t before = app.telemetry().inferences_run;
  const bool flushed = app.FlushInference(t_ms);
  if (flushed) {
    EXPECT_EQ(app.telemetry().inferences_run, before + 1);
  } else {
    // Stride just elapsed on the last block; that Invoke is already the last window.
    EXPECT_GT(inferences, 0u);
    EXPECT_EQ(app.telemetry().inferences_run, before);
  }
  EXPECT_GE(app.last_label(), 0);
}

// Nothing may allocate after Init(). A crude but effective proxy is
// that a long run does not change the process's steady-state behaviour; the
// stronger guarantee comes from the code containing no new/malloc at all.
TEST(KwsApp, RepeatedInitIsIdempotent) {
  KwsApp app;
  ASSERT_TRUE(app.Init());
  auto audio = Tone(KWS_SAMPLE_RATE_HZ * 2, 440.0f);
  Stream(&app, audio);
  ASSERT_TRUE(app.Init());
  EXPECT_EQ(app.telemetry().inferences_run, 0u);
  EXPECT_EQ(app.telemetry().frontend_frames_generated, 0u);
}
