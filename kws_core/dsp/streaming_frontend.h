// SPDX-License-Identifier: Apache-2.0
// Incremental MFCC frontend.
//
// Matches the frontend the model was trained with (tf.signal MFCC):
//
//   x            int16 PCM
//   x / max(x)   peak normalisation over the whole 1 s clip  <-- NOT causal
//   |STFT|       frame 480, hop 320, fft 512, periodic Hann
//   mel          40 bins, 20..4000 Hz, tf.signal.linear_to_mel_weight_matrix
//   log(m+1e-6)
//   DCT-II       tf.signal.mfccs_from_log_mel_spectrograms, first 10 coeffs
//
// Split of work: PushHop() does the per-frame part that is causal and
// expensive (window, FFT, mel) and stores mel energies in the feature ring.
// ComputeMfccWindow() does the part that depends on the clip-wide peak (log,
// DCT) once per inference over 49 frames. The FFT is never recomputed for a
// frame that has already been seen.
#ifndef STREAMING_FRONTEND_H_
#define STREAMING_FRONTEND_H_

#include <stddef.h>
#include <stdint.h>

#include "config/kws_config.h"
#include "dsp/feature_ring_buffer.h"

namespace kws {

class StreamingFrontend {
 public:
  bool Init();
  void Reset();

  // Consumes exactly one hop of new audio. `frame` must hold the most recent
  // KWS_FRAME_LENGTH samples (i.e. the analysis frame ending at the newest
  // sample). Produces one mel-energy frame into `ring`.
  void PushFrame(const AudioSample* frame, FeatureRingBuffer* ring);

  // Applies the clip-wide peak normaliser, log and DCT to the newest
  // KWS_NUM_FRAMES mel frames. `clip_peak` is ClipPeak() over the same 1 s
  // window -- the SIGNED maximum, matching training tf.reduce_max.
  // Writes KWS_NUM_FRAMES * KWS_NUM_MFCC floats, frame-major.
  // Returns false when the ring does not yet hold a full window.
  bool ComputeMfccWindow(const FeatureRingBuffer& ring, float clip_peak, float* out) const;

  // Non-streaming path, used only by the golden test to prove the
  // incremental path agrees with a straight full-window computation.
  bool ComputeMfccWindowDirect(const AudioSample* clip, size_t clip_samples, float* out) const;

  uint32_t frames_generated() const { return frames_generated_; }

  // Exposed for unit tests.
  const float* mel_weights() const { return &mel_weights_[0][0]; }
  const float* window() const { return window_; }

 private:
  void AnalyseFrame(const AudioSample* frame, float* mel_out) const;

  float window_[KWS_FRAME_LENGTH] = {};                    // periodic Hann
  float mel_weights_[KWS_SPECTRUM_BINS][KWS_NUM_MEL] = {};  // linear->mel
  float dct_[KWS_NUM_MFCC][KWS_NUM_MEL] = {};              // DCT-II, TF scaling
  uint32_t frames_generated_ = 0;
  bool initialised_ = false;
};

// Signed maximum of a clip -- the divisor the training frontend uses. Exposed
// because the application must compute it over the same window it infers on.
float ClipPeak(const AudioSample* clip, size_t n);

// Real-FFT magnitude helper, exposed so tests can check it independently.
// `in` holds KWS_FFT_SIZE real samples; `mag_out` receives KWS_SPECTRUM_BINS
// magnitudes. Radix-2; on Apollo510 this is the natural place to substitute
// CMSIS-DSP arm_rfft_fast_f32 once numerical equivalence has been re-verified.
void RealFftMagnitude(const float* in, float* mag_out);

}  // namespace kws
#endif  // STREAMING_FRONTEND_H_
