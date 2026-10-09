// SPDX-License-Identifier: Apache-2.0
// Single source of truth for every derived constant.
// Values come from the shipping .tflite and the training frontend
// (tf.signal MFCC, 30/20 ms).
#ifndef KWS_CONFIG_H_
#define KWS_CONFIG_H_

#include <assert.h>   // C11 provides static_assert here; C++11 has it built in
#include <stddef.h>
#include <stdint.h>

// ---------------------------------------------------------------- audio ----
#define KWS_SAMPLE_RATE_HZ      16000
#define KWS_NUM_CHANNELS        1
// Canonical internal representation: mono 16 kHz signed PCM16.
typedef int16_t AudioSample;
#ifdef __cplusplus
// This header stays C-compatible for the platform/ layer, so the typedef is at
// global scope; alias it so C++ callers can spell it kws::AudioSample.
namespace kws { using AudioSample = ::AudioSample; }
#endif

// Feature frame geometry from the training tf.signal MFCC (30/20 ms), not a
// guessed YAML.
#define KWS_FRAME_LENGTH        480   // 30 ms @ 16 kHz  (tf.signal.stft frame_length)
#define KWS_FRAME_STEP          320   // 20 ms @ 16 kHz  (tf.signal.stft frame_step)
#define KWS_FFT_SIZE            512   // fft_length=None -> next pow2 of 480
#define KWS_SPECTRUM_BINS       (KWS_FFT_SIZE / 2 + 1)   // 257
#define KWS_NUM_MEL             40
#define KWS_NUM_MFCC            10
#define KWS_MEL_LOWER_HZ        20.0f
#define KWS_MEL_UPPER_HZ        4000.0f
#define KWS_LOG_EPSILON         1e-6f

#define KWS_CLIP_SAMPLES        16000 // 1000 ms inference window
#define KWS_NUM_FRAMES          49    // 1 + (16000 - 480) / 320

// --------------------------------------------------------------- model -----
#define KWS_NUM_CLASSES         12

// Quantisation parameters. The generated contract is the source of truth when
// a model has been embedded: a Class B re-quantised .tflite updates
// kws_model_contract.h and these macros follow it. Host-test fallbacks match
// the shipping model so unit tests stay meaningful before embed_model.py runs.
#if defined(__has_include)
#if __has_include("model/generated/kws_model_contract.h")
#define KWS_HAVE_MODEL_CONTRACT 1
#include "model/generated/kws_model_contract.h"
#endif
#endif

#ifdef KWS_HAVE_MODEL_CONTRACT
#if !defined(KWS_MODEL_INPUT_TYPE_INT8) || !defined(KWS_MODEL_OUTPUT_TYPE_INT8)
#error "embedded model is not int8 in/out; model/quantizer.h assumes int8 affine I/O"
#endif
#define KWS_INPUT_SCALE         KWS_MODEL_INPUT_SCALE
#define KWS_INPUT_ZERO_POINT    KWS_MODEL_INPUT_ZERO_POINT
#define KWS_OUTPUT_SCALE        KWS_MODEL_OUTPUT_SCALE
#define KWS_OUTPUT_ZERO_POINT   KWS_MODEL_OUTPUT_ZERO_POINT
static_assert(KWS_NUM_FRAMES * KWS_NUM_MFCC == KWS_MODEL_INPUT_ELEMENTS,
              "frontend window does not fill the model's input tensor");
static_assert(KWS_NUM_CLASSES == KWS_MODEL_OUTPUT_ELEMENTS,
              "label table does not match the model's output width");
#else
#define KWS_INPUT_SCALE         0.5899888277053833f
#define KWS_INPUT_ZERO_POINT    81
#define KWS_OUTPUT_SCALE        0.20843878388404846f
#define KWS_OUTPUT_ZERO_POINT   42
#endif

// ------------------------------------------------------- DMA / cadence -----
// One DMA block == one frame hop, so a completed block yields exactly one
// new feature frame and the frontend never has to buffer a partial hop.
#define KWS_AUDIO_BLOCK_SAMPLES KWS_FRAME_STEP            // 320 = 20 ms
#define KWS_AUDIO_BLOCK_MS      (KWS_AUDIO_BLOCK_SAMPLES * 1000 / KWS_SAMPLE_RATE_HZ)

// PCM ring: inference window + frontend overlap + safety margin.
#define KWS_PCM_RING_SAMPLES    (KWS_CLIP_SAMPLES + KWS_FRAME_LENGTH + 4 * KWS_AUDIO_BLOCK_SAMPLES)

// Feature ring holds one full window plus margin.
#define KWS_FEATURE_RING_FRAMES (KWS_NUM_FRAMES + 8)

// Sliding-window inference cadence: decoupled from frame cadence.
#define KWS_INFERENCE_STRIDE_FRAMES 5                     // 5 * 20 ms = 100 ms

// ------------------------------------------------------- recognizer --------
#define KWS_SMOOTHING_WINDOW    3     // inferences averaged before thresholding
#define KWS_DETECTION_THRESHOLD 0.70f
#define KWS_SUPPRESSION_MS      750
#define KWS_MIN_CONSISTENT      2

// ------------------------------------------------------- logging ----------
#define KWS_LOG_LEVEL_NONE  0
#define KWS_LOG_LEVEL_ERROR 1
#define KWS_LOG_LEVEL_INFO  2
#define KWS_LOG_LEVEL_DEBUG 3
#ifndef KWS_LOG_LEVEL
#define KWS_LOG_LEVEL KWS_LOG_LEVEL_INFO
#endif

// Compile-time contract checks.
// tf.signal.stft emits floor((N - frame_length) / frame_step) + 1 frames.
static_assert(KWS_NUM_FRAMES == 1 + (KWS_CLIP_SAMPLES - KWS_FRAME_LENGTH) / KWS_FRAME_STEP,
              "frame count inconsistent with tf.signal.stft framing");
static_assert(KWS_AUDIO_BLOCK_SAMPLES == KWS_FRAME_STEP,
              "block size must equal hop so one block yields exactly one frame");
static_assert(KWS_PCM_RING_SAMPLES > KWS_CLIP_SAMPLES + KWS_FRAME_LENGTH,
              "PCM ring cannot hold a full inference window plus overlap");
static_assert(KWS_FEATURE_RING_FRAMES > KWS_NUM_FRAMES,
              "feature ring cannot hold a full model window");
static_assert(KWS_FFT_SIZE >= KWS_FRAME_LENGTH, "FFT size must cover the frame");

#endif  // KWS_CONFIG_H_
