// SPDX-License-Identifier: Apache-2.0
// Golden frontend validation.
//
// Host vectors in tests/golden/ were produced with the same tf.signal MFCC
// the model was trained with. On-target MFCC identity is the flashed binary.
#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <algorithm>
#include <string>
#include <vector>

#include "config/kws_config.h"
#include "dsp/feature_ring_buffer.h"
#include "dsp/streaming_frontend.h"
#include "model/quantizer.h"

using kws::AudioSample;
using kws::FeatureRingBuffer;
using kws::StreamingFrontend;

namespace {

struct Golden {
  std::string name;
  std::vector<AudioSample> pcm;
  std::vector<float> features;  // n_frames * n_coeff, frame-major
  int frames = 0;
  int coeff = 0;
};

bool LoadGolden(const std::string& name, Golden* g) {
  const std::string path = std::string(KWS_GOLDEN_DIR) + "/" + name + ".txt";
  std::ifstream in(path);
  if (!in) return false;
  int n_samples = 0;
  in >> g->name >> n_samples >> g->frames >> g->coeff;
  if (!in || n_samples <= 0) return false;
  g->pcm.resize(n_samples);
  for (int i = 0; i < n_samples; ++i) {
    int v = 0;
    in >> v;
    g->pcm[i] = static_cast<AudioSample>(v);
  }
  g->features.resize(static_cast<size_t>(g->frames) * g->coeff);
  for (size_t i = 0; i < g->features.size(); ++i) in >> g->features[i];
  return static_cast<bool>(in);
}

struct ErrStats {
  double max_abs = 0.0;
  double mae = 0.0;
  double rmse = 0.0;
};

ErrStats Compare(const std::vector<float>& got, const std::vector<float>& ref) {
  ErrStats s;
  double sum_abs = 0.0, sum_sq = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double d = static_cast<double>(got[i]) - static_cast<double>(ref[i]);
    s.max_abs = std::max(s.max_abs, std::fabs(d));
    sum_abs += std::fabs(d);
    sum_sq += d * d;
  }
  s.mae = sum_abs / ref.size();
  s.rmse = std::sqrt(sum_sq / ref.size());
  return s;
}

// Drives the incremental path hop by hop, the way the firmware main loop does.
void RunStreaming(const std::vector<AudioSample>& clip, std::vector<float>* out) {
  StreamingFrontend fe;
  ASSERT_TRUE(fe.Init());
  FeatureRingBuffer ring;
  ring.Reset();

  const float peak = kws::ClipPeak(clip.data(), clip.size());

  for (int f = 0; f < KWS_NUM_FRAMES; ++f) {
    AudioSample frame[KWS_FRAME_LENGTH];
    for (int i = 0; i < KWS_FRAME_LENGTH; ++i) frame[i] = clip[f * KWS_FRAME_STEP + i];
    fe.PushFrame(frame, &ring);
  }
  out->assign(KWS_NUM_FRAMES * KWS_NUM_MFCC, 0.0f);
  ASSERT_TRUE(fe.ComputeMfccWindow(ring, peak, out->data()));
}

class GoldenFrontend : public ::testing::TestWithParam<const char*> {};

}  // namespace

// Float tolerance. These bounds are measured, not chosen for convenience.
//
// For speech-like clips the firmware matches the training frontend to ~4e-5.
// The loose end of the range comes from a single mechanism: log(m + 1e-6)
// amplifies relative error when a normalised mel energy approaches the epsilon.
// A pure sweeping tone ("chirp") is the adversarial case -- most mel bands are
// nearly empty at any instant, bottoming out around 1e-4, only ~100x epsilon.
//
// That regime is a property of the reference, not of this implementation: an
// independent float64 NumPy computation of the same pipeline diverges from
// TF's own float32 output by 7.8e-3 max on chirp, versus 9.6e-6 on the
// speech-like clip. So these bounds admit float32 noise in low-energy bands
// and nothing else. The strict gate is the int8 test below -- that is the
// tensor the model actually consumes.
constexpr double kMaxAbsTol = 5e-2;
constexpr double kRmseTol = 1e-2;
// Speech-like clips must stay far tighter than the adversarial bound.
constexpr double kSpeechMaxAbsTol = 5e-4;

TEST_P(GoldenFrontend, DirectPathMatchesTrainingFrontend) {
  Golden g;
  ASSERT_TRUE(LoadGolden(GetParam(), &g))
      << "missing golden vector; run host/gen_golden.py";
  ASSERT_EQ(g.frames, KWS_NUM_FRAMES);
  ASSERT_EQ(g.coeff, KWS_NUM_MFCC);

  StreamingFrontend fe;
  ASSERT_TRUE(fe.Init());
  std::vector<float> got(g.features.size());
  ASSERT_TRUE(fe.ComputeMfccWindowDirect(g.pcm.data(), g.pcm.size(), got.data()));

  const ErrStats s = Compare(got, g.features);
  std::cout << "  [" << GetParam() << "] direct    max_abs=" << s.max_abs
            << " mae=" << s.mae << " rmse=" << s.rmse << "\n";
  EXPECT_LT(s.max_abs, kMaxAbsTol);
  EXPECT_LT(s.rmse, kRmseTol);
  // "chirp" is the deliberately adversarial clip; everything else is either
  // speech-like or trivial and must hold the tight bound.
  if (std::string(GetParam()) != "chirp") {
    EXPECT_LT(s.max_abs, kSpeechMaxAbsTol);
  }
}

TEST_P(GoldenFrontend, StreamingPathMatchesTrainingFrontend) {
  Golden g;
  ASSERT_TRUE(LoadGolden(GetParam(), &g))
      << "missing golden vector; run host/gen_golden.py";

  std::vector<float> got;
  RunStreaming(g.pcm, &got);

  const ErrStats s = Compare(got, g.features);
  std::cout << "  [" << GetParam() << "] streaming max_abs=" << s.max_abs
            << " mae=" << s.mae << " rmse=" << s.rmse << "\n";
  EXPECT_LT(s.max_abs, kMaxAbsTol);
  EXPECT_LT(s.rmse, kRmseTol);
  if (std::string(GetParam()) != "chirp") {
    EXPECT_LT(s.max_abs, kSpeechMaxAbsTol);
  }
}

// Exact-match rate on the quantized features (what the model consumes).
TEST_P(GoldenFrontend, QuantizedFeaturesMatchAlmostExactly) {
  Golden g;
  ASSERT_TRUE(LoadGolden(GetParam(), &g));

  std::vector<float> got;
  RunStreaming(g.pcm, &got);

  size_t exact = 0;
  int worst = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const int a = kws::QuantizeInt8(got[i], KWS_INPUT_SCALE, KWS_INPUT_ZERO_POINT);
    const int b = kws::QuantizeInt8(g.features[i], KWS_INPUT_SCALE, KWS_INPUT_ZERO_POINT);
    if (a == b) ++exact;
    worst = std::max(worst, std::abs(a - b));
  }
  const double pct = 100.0 * static_cast<double>(exact) / static_cast<double>(got.size());
  std::cout << "  [" << GetParam() << "] int8 exact=" << pct << "%  worst_delta=" << worst
            << " LSB\n";
  EXPECT_GE(pct, 99.0);
  EXPECT_LE(worst, 1);
}

INSTANTIATE_TEST_SUITE_P(Corpus, GoldenFrontend,
                         ::testing::Values("synthetic", "synthetic_quiet", "chirp", "impulse",
                                           "silence"),
                         [](const ::testing::TestParamInfo<const char*>& i) {
                           return std::string(i.param);
                         });
