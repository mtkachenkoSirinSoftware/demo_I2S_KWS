// SPDX-License-Identifier: Apache-2.0
#include "dsp/streaming_frontend.h"

#include <math.h>
#include <string.h>

namespace kws {
namespace {

constexpr float kPi = 3.14159265358979323846f;
// tf.signal mel constants (_MEL_HIGH_FREQUENCY_Q, _MEL_BREAK_FREQUENCY_HERTZ).
constexpr float kMelQ = 1127.0f;
constexpr float kMelBreakHz = 700.0f;

inline float HertzToMel(float hz) { return kMelQ * logf(1.0f + hz / kMelBreakHz); }

// In-place iterative radix-2 complex FFT, decimation in time.
void Fft(float* re, float* im, size_t n) {
  for (size_t i = 1, j = 0; i < n; ++i) {
    size_t bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      float tr = re[i]; re[i] = re[j]; re[j] = tr;
      float ti = im[i]; im[i] = im[j]; im[j] = ti;
    }
  }
  for (size_t len = 2; len <= n; len <<= 1) {
    const float ang = -2.0f * kPi / static_cast<float>(len);
    const float wr = cosf(ang), wi = sinf(ang);
    for (size_t i = 0; i < n; i += len) {
      float cr = 1.0f, ci = 0.0f;
      for (size_t k = 0; k < len / 2; ++k) {
        const size_t a = i + k, b = i + k + len / 2;
        const float xr = re[b] * cr - im[b] * ci;
        const float xi = re[b] * ci + im[b] * cr;
        re[b] = re[a] - xr; im[b] = im[a] - xi;
        re[a] += xr;        im[a] += xi;
        const float ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }
}

}  // namespace

float ClipPeak(const AudioSample* clip, size_t n) {
  if (clip == nullptr || n == 0) return 0.0f;
  float peak = static_cast<float>(clip[0]);
  for (size_t i = 1; i < n; ++i) {
    const float v = static_cast<float>(clip[i]);
    if (v > peak) peak = v;
  }
  return peak;
}

void RealFftMagnitude(const float* in, float* mag_out) {
  float re[KWS_FFT_SIZE], im[KWS_FFT_SIZE];
  for (size_t i = 0; i < KWS_FFT_SIZE; ++i) { re[i] = in[i]; im[i] = 0.0f; }
  Fft(re, im, KWS_FFT_SIZE);
  for (size_t k = 0; k < KWS_SPECTRUM_BINS; ++k) {
    mag_out[k] = sqrtf(re[k] * re[k] + im[k] * im[k]);
  }
}

bool StreamingFrontend::Init() {
  // Periodic Hann, matching tf.signal.hann_window(periodic=True).
  for (size_t n = 0; n < KWS_FRAME_LENGTH; ++n) {
    window_[n] = 0.5f - 0.5f * cosf(2.0f * kPi * static_cast<float>(n) /
                                    static_cast<float>(KWS_FRAME_LENGTH));
  }

  // tf.signal.linear_to_mel_weight_matrix. Bin 0 (DC) is deliberately zeroed:
  // TF slices it off via bands_to_zero=1.
  memset(mel_weights_, 0, sizeof(mel_weights_));
  const float nyquist = static_cast<float>(KWS_SAMPLE_RATE_HZ) / 2.0f;
  const float mel_lo = HertzToMel(KWS_MEL_LOWER_HZ);
  const float mel_hi = HertzToMel(KWS_MEL_UPPER_HZ);
  float edges[KWS_NUM_MEL + 2];
  for (size_t i = 0; i < KWS_NUM_MEL + 2; ++i) {
    edges[i] = mel_lo + (mel_hi - mel_lo) * static_cast<float>(i) /
                            static_cast<float>(KWS_NUM_MEL + 1);
  }
  for (size_t bin = 1; bin < KWS_SPECTRUM_BINS; ++bin) {
    const float hz = nyquist * static_cast<float>(bin) /
                     static_cast<float>(KWS_SPECTRUM_BINS - 1);
    const float mel = HertzToMel(hz);
    for (size_t m = 0; m < KWS_NUM_MEL; ++m) {
      const float lower = edges[m], center = edges[m + 1], upper = edges[m + 2];
      const float lo_slope = (mel - lower) / (center - lower);
      const float hi_slope = (upper - mel) / (upper - center);
      float w = lo_slope < hi_slope ? lo_slope : hi_slope;
      if (w < 0.0f) w = 0.0f;
      mel_weights_[bin][m] = w;
    }
  }

  // tf.signal.mfccs_from_log_mel_spectrograms:
  //   dct2[k] = 2 * sum_n x[n] * cos(pi*k*(2n+1)/(2N)),  then * rsqrt(2N).
  const float scale = 2.0f / sqrtf(2.0f * static_cast<float>(KWS_NUM_MEL));
  for (size_t k = 0; k < KWS_NUM_MFCC; ++k) {
    for (size_t n = 0; n < KWS_NUM_MEL; ++n) {
      dct_[k][n] = scale * cosf(kPi * static_cast<float>(k) *
                                (2.0f * static_cast<float>(n) + 1.0f) /
                                (2.0f * static_cast<float>(KWS_NUM_MEL)));
    }
  }
  initialised_ = true;
  frames_generated_ = 0;
  return true;
}

void StreamingFrontend::Reset() { frames_generated_ = 0; }

void StreamingFrontend::AnalyseFrame(const AudioSample* frame, float* mel_out) const {
  float buf[KWS_FFT_SIZE];
  // Window the 480 real samples, zero-pad to 512 exactly as tf.signal.stft does.
  for (size_t i = 0; i < KWS_FRAME_LENGTH; ++i) {
    buf[i] = static_cast<float>(frame[i]) * window_[i];
  }
  for (size_t i = KWS_FRAME_LENGTH; i < KWS_FFT_SIZE; ++i) buf[i] = 0.0f;

  float mag[KWS_SPECTRUM_BINS];
  RealFftMagnitude(buf, mag);

  for (size_t m = 0; m < KWS_NUM_MEL; ++m) mel_out[m] = 0.0f;
  for (size_t bin = 1; bin < KWS_SPECTRUM_BINS; ++bin) {
    const float v = mag[bin];
    if (v == 0.0f) continue;
    for (size_t m = 0; m < KWS_NUM_MEL; ++m) mel_out[m] += v * mel_weights_[bin][m];
  }
}

void StreamingFrontend::PushFrame(const AudioSample* frame, FeatureRingBuffer* ring) {
  if (!initialised_ || frame == nullptr || ring == nullptr) return;
  float mel[KWS_NUM_MEL];
  AnalyseFrame(frame, mel);
  ring->PushFrame(mel);
  ++frames_generated_;
}

bool StreamingFrontend::ComputeMfccWindow(const FeatureRingBuffer& ring, float clip_peak,
                                          float* out) const {
  if (!initialised_ || out == nullptr || !ring.WindowReady()) return false;
  // The training frontend divides by tf.reduce_max(x) -- the SIGNED maximum, not max|x|.
  // The magnitude spectrum is insensitive to the divisor's sign, so only its
  // magnitude matters here, but which sample supplies it does matter: for a
  // clip whose negative excursion exceeds its positive one the two differ, and
  // the resulting scale error lands entirely in MFCC coefficient 0.
  // A zero peak is left unscaled, matching the reference's guarded divide.
  const float inv_peak = (clip_peak == 0.0f) ? 1.0f : (1.0f / fabsf(clip_peak));

  for (size_t f = 0; f < KWS_NUM_FRAMES; ++f) {
    const float* mel = ring.WindowFrame(f);
    float log_mel[KWS_NUM_MEL];
    for (size_t m = 0; m < KWS_NUM_MEL; ++m) {
      log_mel[m] = logf(mel[m] * inv_peak + KWS_LOG_EPSILON);
    }
    float* dst = out + f * KWS_NUM_MFCC;
    for (size_t k = 0; k < KWS_NUM_MFCC; ++k) {
      float acc = 0.0f;
      for (size_t m = 0; m < KWS_NUM_MEL; ++m) acc += log_mel[m] * dct_[k][m];
      dst[k] = acc;
    }
  }
  return true;
}

bool StreamingFrontend::ComputeMfccWindowDirect(const AudioSample* clip, size_t clip_samples,
                                                float* out) const {
  if (!initialised_ || clip == nullptr || out == nullptr) return false;
  if (clip_samples < KWS_CLIP_SAMPLES) return false;

  const float peak = ClipPeak(clip, KWS_CLIP_SAMPLES);
  const float inv_peak = (peak == 0.0f) ? 1.0f : (1.0f / fabsf(peak));

  for (size_t f = 0; f < KWS_NUM_FRAMES; ++f) {
    float mel[KWS_NUM_MEL];
    AnalyseFrame(clip + f * KWS_FRAME_STEP, mel);
    float log_mel[KWS_NUM_MEL];
    for (size_t m = 0; m < KWS_NUM_MEL; ++m) {
      log_mel[m] = logf(mel[m] * inv_peak + KWS_LOG_EPSILON);
    }
    float* dst = out + f * KWS_NUM_MFCC;
    for (size_t k = 0; k < KWS_NUM_MFCC; ++k) {
      float acc = 0.0f;
      for (size_t m = 0; m < KWS_NUM_MEL; ++m) acc += log_mel[m] * dct_[k][m];
      dst[k] = acc;
    }
  }
  return true;
}

}  // namespace kws
