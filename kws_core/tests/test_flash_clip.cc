// SPDX-License-Identifier: Apache-2.0
// Flash-clip input identity: player PCM == golden, main-loop drain
// still hits inference stride, StreamEnd flushes the last window.
#include "audio/flash_clip_player.h"

#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "app/kws_app.h"
#include "config/kws_config.h"
#include "dsp/feature_ring_buffer.h"
#include "dsp/streaming_frontend.h"
#include "model/quantizer.h"

using kws::KwsApp;

namespace {

bool LoadGoldenPcm(const std::string& name, std::vector<AudioSample>* pcm,
                   std::vector<float>* features) {
  const std::string path = std::string(KWS_GOLDEN_DIR) + "/" + name + ".txt";
  std::ifstream in(path);
  if (!in) return false;
  std::string tag;
  int n_samples = 0, frames = 0, coeff = 0;
  in >> tag >> n_samples >> frames >> coeff;
  if (!in || n_samples <= 0) return false;
  pcm->resize(static_cast<size_t>(n_samples));
  for (int i = 0; i < n_samples; ++i) {
    int v = 0;
    in >> v;
    (*pcm)[static_cast<size_t>(i)] = static_cast<AudioSample>(v);
  }
  if (features != nullptr) {
    features->resize(static_cast<size_t>(frames) * static_cast<size_t>(coeff));
    for (float& x : *features) in >> x;
  }
  return static_cast<bool>(in);
}

struct MainLoopResult {
  std::vector<AudioSample> emitted;
  uint32_t inferences = 0;
  bool saw_begin = false;
  bool saw_end = false;
  bool ok = true;
};

MainLoopResult RunAsMain(FlashClipPlayer* player, KwsApp* app) {
  MainLoopResult r;
  if (!FlashClipPlayerStart(player)) {
    r.ok = false;
    return r;
  }
  for (int spins = 0; spins < 256; ++spins) {
    unsigned drained = 0;
    while (FlashClipPlayerBlockReady(player) && drained < 8) {
      uint32_t count = 0;
      const AudioSample* block = FlashClipPlayerAcquireBlock(player, &count);
      if (block == nullptr) {
        r.ok = false;
        return r;
      }
      r.emitted.insert(r.emitted.end(), block, block + count);
      app->OnAudioBlock(block, count, 0);
      FlashClipPlayerReleaseBlock(player);
      ++drained;
      if (app->InferenceDue()) {
        app->RunInference(0);
        ++r.inferences;
      }
    }
    if (FlashClipPlayerTakeStreamBegin(player)) r.saw_begin = true;
    if (app->InferenceDue()) {
      app->RunInference(0);
      ++r.inferences;
    }
    if (FlashClipPlayerTakeStreamEnd(player)) {
      r.saw_end = true;
      if (app->FlushInference(0)) ++r.inferences;
      break;
    }
  }
  return r;
}

}  // namespace

TEST(FlashClip, EmitsGoldenPcmHopByHop) {
  std::vector<AudioSample> pcm;
  ASSERT_TRUE(LoadGoldenPcm("synthetic", &pcm, nullptr));
  ASSERT_EQ(pcm.size(), static_cast<size_t>(KWS_CLIP_SAMPLES));

  FlashClipPlayer player{};
  ASSERT_TRUE(FlashClipPlayerInit(&player, pcm.data(), static_cast<uint32_t>(pcm.size())));
  ASSERT_TRUE(FlashClipPlayerStart(&player));
  EXPECT_TRUE(FlashClipPlayerTakeStreamBegin(&player));

  std::vector<AudioSample> got;
  while (FlashClipPlayerBlockReady(&player)) {
    uint32_t n = 0;
    const AudioSample* b = FlashClipPlayerAcquireBlock(&player, &n);
    ASSERT_NE(b, nullptr);
    EXPECT_EQ(n, static_cast<uint32_t>(KWS_AUDIO_BLOCK_SAMPLES));
    got.insert(got.end(), b, b + n);
    FlashClipPlayerReleaseBlock(&player);
  }
  EXPECT_TRUE(FlashClipPlayerTakeStreamEnd(&player));
  ASSERT_EQ(got.size(), pcm.size());
  EXPECT_EQ(got, pcm);
}

TEST(FlashClip, MainLoopDoesNotSkipStrideAndFlushes) {
  std::vector<AudioSample> pcm;
  ASSERT_TRUE(LoadGoldenPcm("synthetic", &pcm, nullptr));

  FlashClipPlayer player{};
  ASSERT_TRUE(FlashClipPlayerInit(&player, pcm.data(), static_cast<uint32_t>(pcm.size())));
  KwsApp app;
  ASSERT_TRUE(app.Init());
  const MainLoopResult r = RunAsMain(&player, &app);

  EXPECT_TRUE(r.ok);
  EXPECT_TRUE(r.saw_begin);
  EXPECT_TRUE(r.saw_end);
  ASSERT_EQ(r.emitted.size(), pcm.size());
  EXPECT_EQ(r.emitted, pcm);
  EXPECT_GE(r.inferences, 1u);
  EXPECT_EQ(app.telemetry().audio_overruns, 0u);
  EXPECT_GE(app.last_label(), 0);
}

TEST(FlashClip, InnerInferenceDueKeepsStrideOnTwoSecondClip) {
  std::vector<AudioSample> one;
  ASSERT_TRUE(LoadGoldenPcm("synthetic", &one, nullptr));
  std::vector<AudioSample> pcm = one;
  pcm.insert(pcm.end(), one.begin(), one.end());

  FlashClipPlayer player{};
  ASSERT_TRUE(FlashClipPlayerInit(&player, pcm.data(), static_cast<uint32_t>(pcm.size())));
  KwsApp app;
  ASSERT_TRUE(app.Init());
  const MainLoopResult r = RunAsMain(&player, &app);
  EXPECT_TRUE(r.ok);
  EXPECT_TRUE(r.saw_end);
  ASSERT_EQ(r.emitted.size(), pcm.size());
  // Window ready at 49 frames; then every KWS_INFERENCE_STRIDE_FRAMES.
  // 2 s → 99 frames → (99 - 49) / 5 + 1 = 11 inferences if stride is honoured.
  EXPECT_GE(r.inferences, 8u);
}

TEST(FlashClip, LastWindowMatchesTrainingFrontendInt8) {
  std::vector<AudioSample> pcm;
  std::vector<float> ref;
  ASSERT_TRUE(LoadGoldenPcm("synthetic", &pcm, &ref));

  FlashClipPlayer player{};
  ASSERT_TRUE(FlashClipPlayerInit(&player, pcm.data(), static_cast<uint32_t>(pcm.size())));
  std::vector<AudioSample> got;
  ASSERT_TRUE(FlashClipPlayerStart(&player));
  while (FlashClipPlayerBlockReady(&player)) {
    uint32_t n = 0;
    const AudioSample* b = FlashClipPlayerAcquireBlock(&player, &n);
    got.insert(got.end(), b, b + n);
    FlashClipPlayerReleaseBlock(&player);
  }

  kws::StreamingFrontend fe;
  ASSERT_TRUE(fe.Init());
  kws::FeatureRingBuffer ring;
  ring.Reset();
  const float peak = kws::ClipPeak(got.data(), got.size());
  for (int f = 0; f < KWS_NUM_FRAMES; ++f) {
    AudioSample frame[KWS_FRAME_LENGTH];
    for (int i = 0; i < KWS_FRAME_LENGTH; ++i) {
      frame[i] = got[static_cast<size_t>(f * KWS_FRAME_STEP + i)];
    }
    fe.PushFrame(frame, &ring);
  }
  std::vector<float> mfcc(static_cast<size_t>(KWS_NUM_FRAMES * KWS_NUM_MFCC));
  ASSERT_TRUE(fe.ComputeMfccWindow(ring, peak, mfcc.data()));

  size_t exact = 0;
  int worst = 0;
  for (size_t i = 0; i < mfcc.size(); ++i) {
    const int a = kws::QuantizeInt8(mfcc[i], KWS_INPUT_SCALE, KWS_INPUT_ZERO_POINT);
    const int b = kws::QuantizeInt8(ref[i], KWS_INPUT_SCALE, KWS_INPUT_ZERO_POINT);
    if (a == b) ++exact;
    const int d = a > b ? a - b : b - a;
    if (d > worst) worst = d;
  }
  const double pct = 100.0 * static_cast<double>(exact) / static_cast<double>(mfcc.size());
  EXPECT_GE(pct, 99.0);
  EXPECT_LE(worst, 1);
}
