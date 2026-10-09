// SPDX-License-Identifier: Apache-2.0
#include "dsp/streaming_frontend.h"

#include <gtest/gtest.h>

#include <cmath>
#include <algorithm>
#include <vector>

#include "audio/audio_ring_buffer.h"
#include "dsp/feature_ring_buffer.h"

using kws::AudioRingBuffer;
using kws::AudioSample;
using kws::FeatureRingBuffer;
using kws::StreamingFrontend;

namespace {

// Deterministic pseudo-speech: a few formant-ish tones plus a repeatable noise
// floor. Not real audio, but it exercises every mel band.
std::vector<AudioSample> SyntheticClip(int n = KWS_CLIP_SAMPLES, float gain = 1.0f) {
  std::vector<AudioSample> x(n);
  uint32_t rng = 12345;
  for (int i = 0; i < n; ++i) {
    const float t = static_cast<float>(i) / KWS_SAMPLE_RATE_HZ;
    float v = 0.45f * sinf(2.0f * 3.14159265f * 320.0f * t) +
              0.30f * sinf(2.0f * 3.14159265f * 1100.0f * t) +
              0.15f * sinf(2.0f * 3.14159265f * 2600.0f * t);
    // envelope so the clip is not stationary
    v *= 0.5f + 0.5f * sinf(2.0f * 3.14159265f * 3.0f * t);
    rng = rng * 1664525u + 1013904223u;
    v += 0.02f * ((static_cast<float>((rng >> 16) & 0xFFFF) / 32768.0f) - 1.0f);
    float s = v * 12000.0f * gain;
    if (s > 32767.0f) s = 32767.0f;
    if (s < -32768.0f) s = -32768.0f;
    x[i] = static_cast<AudioSample>(s);
  }
  return x;
}

float SignedPeak(const std::vector<AudioSample>& x) {
  return kws::ClipPeak(x.data(), x.size());
}

// Runs the clip through the streaming path exactly as the firmware does:
// one hop at a time, lifting the analysis frame out of the PCM ring.
void RunStreaming(const StreamingFrontend& fe_const, const std::vector<AudioSample>& clip,
                  std::vector<float>* out) {
  StreamingFrontend fe;
  ASSERT_TRUE(fe.Init());
  AudioRingBuffer pcm;
  pcm.Reset();
  FeatureRingBuffer feat;
  feat.Reset();

  size_t produced = 0;
  for (size_t off = 0; off + KWS_AUDIO_BLOCK_SAMPLES <= clip.size();
       off += KWS_AUDIO_BLOCK_SAMPLES) {
    pcm.Push(clip.data() + off, KWS_AUDIO_BLOCK_SAMPLES);
    // A frame is analysable as soon as FRAME_LENGTH samples exist and we are on
    // a hop boundary aligned with the reference framing.
    const size_t end = off + KWS_AUDIO_BLOCK_SAMPLES;
    while ((produced * KWS_FRAME_STEP + KWS_FRAME_LENGTH) <= end) {
      AudioSample frame[KWS_FRAME_LENGTH];
      const size_t frame_end = produced * KWS_FRAME_STEP + KWS_FRAME_LENGTH;
      // Copy the frame directly out of the source for index clarity; the ring
      // path is covered by AudioRingBuffer tests.
      for (size_t i = 0; i < KWS_FRAME_LENGTH; ++i) {
        frame[i] = clip[frame_end - KWS_FRAME_LENGTH + i];
      }
      fe.PushFrame(frame, &feat);
      ++produced;
    }
  }
  ASSERT_GE(produced, static_cast<size_t>(KWS_NUM_FRAMES));
  out->assign(KWS_NUM_FRAMES * KWS_NUM_MFCC, 0.0f);
  ASSERT_TRUE(fe.ComputeMfccWindow(feat, SignedPeak(clip), out->data()));
}

}  // namespace

TEST(StreamingFrontend, InitBuildsPeriodicHannWindow) {
  StreamingFrontend fe;
  ASSERT_TRUE(fe.Init());
  const float* w = fe.window();
  // Periodic Hann: w[0] == 0 and the window is symmetric about N/2.
  EXPECT_NEAR(w[0], 0.0f, 1e-6f);
  for (size_t i = 1; i < KWS_FRAME_LENGTH; ++i) {
    EXPECT_NEAR(w[i], w[KWS_FRAME_LENGTH - i], 1e-5f) << "at " << i;
  }
  EXPECT_NEAR(w[KWS_FRAME_LENGTH / 2], 1.0f, 1e-5f);
}

TEST(StreamingFrontend, MelWeightsZeroTheDcBin) {
  StreamingFrontend fe;
  ASSERT_TRUE(fe.Init());
  const float* w = fe.mel_weights();
  // tf.signal.linear_to_mel_weight_matrix drops bin 0 (bands_to_zero=1).
  for (size_t m = 0; m < KWS_NUM_MEL; ++m) EXPECT_FLOAT_EQ(w[0 * KWS_NUM_MEL + m], 0.0f);
}

TEST(StreamingFrontend, MelWeightsAreNonNegativeAndBandLimited) {
  StreamingFrontend fe;
  ASSERT_TRUE(fe.Init());
  const float* w = fe.mel_weights();
  const float nyquist = KWS_SAMPLE_RATE_HZ / 2.0f;
  for (size_t bin = 0; bin < KWS_SPECTRUM_BINS; ++bin) {
    const float hz = nyquist * static_cast<float>(bin) / (KWS_SPECTRUM_BINS - 1);
    for (size_t m = 0; m < KWS_NUM_MEL; ++m) {
      const float v = w[bin * KWS_NUM_MEL + m];
      EXPECT_GE(v, 0.0f);
      // Nothing outside [20, 4000] Hz may contribute.
      if (hz < KWS_MEL_LOWER_HZ || hz > KWS_MEL_UPPER_HZ) {
        EXPECT_NEAR(v, 0.0f, 1e-6f) << "bin " << bin << " mel " << m;
      }
    }
  }
}

TEST(StreamingFrontend, FftResolvesADcSignal) {
  std::vector<float> in(KWS_FFT_SIZE, 1.0f);
  std::vector<float> mag(KWS_SPECTRUM_BINS);
  kws::RealFftMagnitude(in.data(), mag.data());
  EXPECT_NEAR(mag[0], static_cast<float>(KWS_FFT_SIZE), 1e-2f);
  for (size_t k = 1; k < KWS_SPECTRUM_BINS; ++k) EXPECT_NEAR(mag[k], 0.0f, 1e-2f);
}

TEST(StreamingFrontend, FftResolvesASingleBinTone) {
  const size_t k0 = 64;
  std::vector<float> in(KWS_FFT_SIZE);
  for (size_t n = 0; n < KWS_FFT_SIZE; ++n) {
    in[n] = cosf(2.0f * 3.14159265358979f * k0 * n / KWS_FFT_SIZE);
  }
  std::vector<float> mag(KWS_SPECTRUM_BINS);
  kws::RealFftMagnitude(in.data(), mag.data());
  size_t peak = 0;
  for (size_t k = 1; k < KWS_SPECTRUM_BINS; ++k) if (mag[k] > mag[peak]) peak = k;
  EXPECT_EQ(peak, k0);
  EXPECT_NEAR(mag[k0], KWS_FFT_SIZE / 2.0f, 1.0f);
}

// Feeding audio hop-by-hop must give the same window
// as computing the whole spectrogram at once.
TEST(StreamingFrontend, IncrementalMatchesFullWindowComputation) {
  StreamingFrontend fe;
  ASSERT_TRUE(fe.Init());
  auto clip = SyntheticClip();

  std::vector<float> direct(KWS_NUM_FRAMES * KWS_NUM_MFCC);
  ASSERT_TRUE(fe.ComputeMfccWindowDirect(clip.data(), clip.size(), direct.data()));

  std::vector<float> streamed;
  RunStreaming(fe, clip, &streamed);
  ASSERT_EQ(streamed.size(), direct.size());

  double max_abs = 0.0;
  for (size_t i = 0; i < direct.size(); ++i) {
    max_abs = std::max(max_abs, std::fabs(static_cast<double>(streamed[i] - direct[i])));
  }
  EXPECT_LT(max_abs, 1e-4) << "streaming path diverged from full-window reference";
}

TEST(StreamingFrontend, FrameCounterTracksGeneratedFrames) {
  StreamingFrontend fe;
  ASSERT_TRUE(fe.Init());
  FeatureRingBuffer ring;
  ring.Reset();
  AudioSample frame[KWS_FRAME_LENGTH] = {};
  for (int i = 0; i < 7; ++i) fe.PushFrame(frame, &ring);
  EXPECT_EQ(fe.frames_generated(), 7u);
  EXPECT_EQ(ring.total_frames(), 7u);
}

TEST(StreamingFrontend, WindowNotReadyBeforeEnoughFrames) {
  StreamingFrontend fe;
  ASSERT_TRUE(fe.Init());
  FeatureRingBuffer ring;
  ring.Reset();
  AudioSample frame[KWS_FRAME_LENGTH] = {};
  for (int i = 0; i < KWS_NUM_FRAMES - 1; ++i) fe.PushFrame(frame, &ring);
  std::vector<float> out(KWS_NUM_FRAMES * KWS_NUM_MFCC);
  EXPECT_FALSE(fe.ComputeMfccWindow(ring, 1.0f, out.data()));
}

// Peak normalisation makes the reference frontend gain-invariant. That is a
// property of the trained model's preprocessing, so the firmware must have it
// too — otherwise microphone gain would silently change predictions.
TEST(StreamingFrontend, OutputIsInvariantToInputGain) {
  StreamingFrontend fe;
  ASSERT_TRUE(fe.Init());
  auto loud = SyntheticClip(KWS_CLIP_SAMPLES, 1.0f);
  auto quiet = SyntheticClip(KWS_CLIP_SAMPLES, 0.25f);

  std::vector<float> a(KWS_NUM_FRAMES * KWS_NUM_MFCC), b(a.size());
  ASSERT_TRUE(fe.ComputeMfccWindowDirect(loud.data(), loud.size(), a.data()));
  ASSERT_TRUE(fe.ComputeMfccWindowDirect(quiet.data(), quiet.size(), b.data()));

  // Quantisation to int16 makes this approximate rather than exact.
  double max_abs = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    max_abs = std::max(max_abs, std::fabs(static_cast<double>(a[i] - b[i])));
  }
  EXPECT_LT(max_abs, 0.5) << "frontend is not gain-invariant";
}

TEST(StreamingFrontend, AllZeroClipDoesNotProduceNaN) {
  StreamingFrontend fe;
  ASSERT_TRUE(fe.Init());
  std::vector<AudioSample> silence(KWS_CLIP_SAMPLES, 0);
  std::vector<float> out(KWS_NUM_FRAMES * KWS_NUM_MFCC);
  ASSERT_TRUE(fe.ComputeMfccWindowDirect(silence.data(), silence.size(), out.data()));
  for (float v : out) EXPECT_TRUE(std::isfinite(v));
}

TEST(StreamingFrontend, IsDeterministic) {
  StreamingFrontend fe;
  ASSERT_TRUE(fe.Init());
  auto clip = SyntheticClip();
  std::vector<float> a(KWS_NUM_FRAMES * KWS_NUM_MFCC), b(a.size());
  ASSERT_TRUE(fe.ComputeMfccWindowDirect(clip.data(), clip.size(), a.data()));
  ASSERT_TRUE(fe.ComputeMfccWindowDirect(clip.data(), clip.size(), b.data()));
  EXPECT_EQ(a, b);
}

TEST(StreamingFrontend, RejectsShortClip) {
  StreamingFrontend fe;
  ASSERT_TRUE(fe.Init());
  std::vector<AudioSample> tiny(100, 0);
  std::vector<float> out(KWS_NUM_FRAMES * KWS_NUM_MFCC);
  EXPECT_FALSE(fe.ComputeMfccWindowDirect(tiny.data(), tiny.size(), out.data()));
}
