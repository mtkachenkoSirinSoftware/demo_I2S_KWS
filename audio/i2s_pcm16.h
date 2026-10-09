// SPDX-License-Identifier: Apache-2.0
// I2S DMA words -> one hop of mono PCM16. Pure C, no HAL: host-testable.
//
// The microphone fills one of the two 32-bit slots of a stereo frame with a
// 24-bit sample in bits [23:0]; the other slot reads zero. Which word of the
// pair carries the sample is not fixed (bring-up saw both), so it is chosen
// per hop from the data.
//
// kws_core divides the 1 s window by its signed maximum, so a DC offset or
// mains hum would set the scale of every feature: a one-pole high-pass runs
// on the 24-bit signal before it is narrowed to int16.
#ifndef I2S_PCM16_H_
#define I2S_PCM16_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KWS_I2S_SAMPLE_BITS 24
#define KWS_I2S_WORDS_PER_FRAME 2

typedef struct {
  float hpf_alpha;         // 1 / (1 + 2*pi*fc/fs); 1.0 disables the filter
  float hpf_prev_in;
  float hpf_prev_out;
  uint32_t shift;          // right shift from 24-bit to int16 (8 = top 16 bits)
  uint32_t startup_left;   // hops still to be muted after start
  uint32_t slot;           // word index (0 or 1) that carries the microphone
  uint32_t slot_known;     // 0 until the first unambiguous hop
  uint32_t slot_changes;   // times the slot moved after it was first known
  uint32_t ambiguous_hops; // hops with data in both slots or in neither
} KwsI2sPcmState;

typedef struct {
  int32_t peak_abs;  // of the int16 hop handed to kws_core
  int32_t dc;        // mean of that hop
  uint32_t slot;
  uint32_t muted;    // 1 while the start-up discard is active
} KwsI2sHopStats;

static inline void KwsI2sPcmInit(KwsI2sPcmState* st, uint32_t sample_rate_hz, uint32_t hpf_hz,
                                 uint32_t shift, uint32_t startup_hops) {
  const float two_pi = 6.28318530718f;
  st->hpf_alpha =
      (hpf_hz == 0u) ? 1.0f : 1.0f / (1.0f + two_pi * (float)hpf_hz / (float)sample_rate_hz);
  st->hpf_prev_in = 0.0f;
  st->hpf_prev_out = 0.0f;
  st->shift = shift;
  st->startup_left = startup_hops;
  st->slot = 0u;
  st->slot_known = 0u;
  st->slot_changes = 0u;
  st->ambiguous_hops = 0u;
}

// Sign-extends the 24-bit sample held in bits [23:0] of a DMA word.
static inline int32_t KwsI2sWordToSample(uint32_t word) {
  return (int32_t)((word & 0x00FFFFFFu) ^ 0x00800000u) - 0x00800000;
}

// Picks the slot for this hop. Returns 1 when the hop is unambiguous.
static inline uint32_t KwsI2sPickSlot(KwsI2sPcmState* st, const uint32_t* words, uint32_t frames) {
  uint32_t nonzero0 = 0u;
  uint32_t nonzero1 = 0u;
  for (uint32_t i = 0; i < frames; ++i) {
    nonzero0 += (words[2u * i] != 0u);
    nonzero1 += (words[2u * i + 1u] != 0u);
  }
  if ((nonzero0 != 0u) == (nonzero1 != 0u)) {
    ++st->ambiguous_hops;
    return 0u;
  }
  const uint32_t slot = (nonzero1 != 0u) ? 1u : 0u;
  if (st->slot_known && slot != st->slot) {
    ++st->slot_changes;
  }
  st->slot = slot;
  st->slot_known = 1u;
  return 1u;
}

// Converts `frames` stereo frames (2 * frames words) into `frames` int16
// samples. The filter always runs; the output is zeroed during start-up.
static inline void KwsI2sHopToPcm16(KwsI2sPcmState* st, const uint32_t* words, uint32_t frames,
                                    int16_t* out, KwsI2sHopStats* stats) {
  (void)KwsI2sPickSlot(st, words, frames);
  const uint32_t muted = (st->startup_left != 0u);
  const float scale = 1.0f / (float)(1u << st->shift);
  int32_t peak = 0;
  int64_t sum = 0;
  for (uint32_t i = 0; i < frames; ++i) {
    const float x = (float)KwsI2sWordToSample(words[2u * i + st->slot]);
    const float y = st->hpf_alpha * (st->hpf_prev_out + x - st->hpf_prev_in);
    st->hpf_prev_in = x;
    st->hpf_prev_out = y;
    float v = y * scale;
    v += (v >= 0.0f) ? 0.5f : -0.5f;
    int32_t s = (int32_t)v;
    if (s > 32767) s = 32767;
    if (s < -32768) s = -32768;
    if (muted) s = 0;
    out[i] = (int16_t)s;
    const int32_t a = (s < 0) ? -s : s;
    if (a > peak) peak = a;
    sum += s;
  }
  if (muted) --st->startup_left;
  if (stats != 0) {
    stats->peak_abs = peak;
    stats->dc = (frames != 0u) ? (int32_t)(sum / (int64_t)frames) : 0;
    stats->slot = st->slot;
    stats->muted = muted;
  }
}

#ifdef __cplusplus
}
#endif
#endif  // I2S_PCM16_H_
