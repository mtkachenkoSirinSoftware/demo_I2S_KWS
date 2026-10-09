// SPDX-License-Identifier: Apache-2.0
// Host-only: I2S geometry + DMA words -> PCM16 hop (no EVB, no heliaRT).
#include "i2s_audio.h"

#include <cmath>
#include <cstdio>
#include <vector>

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

namespace {

constexpr uint32_t kFrames = KWS_I2S_HOP_FRAMES;

// One hop with `sample(i)` in word `slot` of each pair and zero in the other.
template <typename F>
std::vector<uint32_t> MakeHop(uint32_t slot, F sample) {
  std::vector<uint32_t> words(2 * kFrames, 0u);
  for (uint32_t i = 0; i < kFrames; ++i) {
    words[2 * i + slot] = static_cast<uint32_t>(sample(i)) & 0x00FFFFFFu;
  }
  return words;
}

KwsI2sPcmState NewState(uint32_t hpf_hz, uint32_t shift, uint32_t startup_hops) {
  KwsI2sPcmState st;
  KwsI2sPcmInit(&st, KWS_I2S_FS_HZ, hpf_hz, shift, startup_hops);
  return st;
}

}  // namespace

int main() {
  // Geometry: one DMA block is one kws_core hop at the KWS sample rate.
  CHECK(KWS_I2S_FS_HZ == 16000u);
  CHECK(KWS_I2S_BCLK_HZ == 1024000u);
  CHECK(KWS_I2S_HOP_FRAMES == 320u);
  CHECK(KWS_I2S_DMA_WORDS == 640u);
  CHECK(KWS_I2S_SYSPLL_HZ / 8u / 3u == KWS_I2S_BCLK_HZ);

  // 24-bit sign extension from bits [23:0].
  CHECK(KwsI2sWordToSample(0x00000000u) == 0);
  CHECK(KwsI2sWordToSample(0x007FFFFFu) == 8388607);
  CHECK(KwsI2sWordToSample(0x00800000u) == -8388608);
  CHECK(KwsI2sWordToSample(0x00FFFB91u) == -1135);  // word seen on the EVB
  CHECK(KwsI2sWordToSample(0xAB000001u) == 1);      // top byte is ignored

  std::vector<int16_t> out(kFrames);
  KwsI2sHopStats stats;

  // Sample in word 0, filter off: top 16 of 24 bits, rounded.
  {
    KwsI2sPcmState st = NewState(0, 8, 0);
    auto words = MakeHop(0, [](uint32_t i) { return static_cast<int32_t>(i * 256); });
    KwsI2sHopToPcm16(&st, words.data(), kFrames, out.data(), &stats);
    CHECK(st.slot == 0u && st.slot_known == 1u && st.slot_changes == 0u);
    CHECK(out[0] == 0 && out[1] == 1 && out[319] == 319);
    CHECK(stats.peak_abs == 319 && stats.slot == 0u && stats.muted == 0u);
  }

  // Sample in word 1: found without being told, negative values survive.
  {
    KwsI2sPcmState st = NewState(0, 8, 0);
    auto words = MakeHop(1, [](uint32_t i) { return -static_cast<int32_t>(i + 1) * 256; });
    KwsI2sHopToPcm16(&st, words.data(), kFrames, out.data(), &stats);
    CHECK(st.slot == 1u && stats.slot == 1u);
    CHECK(out[0] == -1 && out[319] == -320);
  }

  // The slot moving between hops is followed and counted.
  {
    KwsI2sPcmState st = NewState(0, 8, 0);
    auto a = MakeHop(0, [](uint32_t) { return 512; });
    auto b = MakeHop(1, [](uint32_t) { return 512; });
    KwsI2sHopToPcm16(&st, a.data(), kFrames, out.data(), &stats);
    KwsI2sHopToPcm16(&st, b.data(), kFrames, out.data(), &stats);
    CHECK(st.slot == 1u && st.slot_changes == 1u);
    CHECK(out[10] == 2);
    KwsI2sHopToPcm16(&st, b.data(), kFrames, out.data(), &stats);
    CHECK(st.slot_changes == 1u);
  }

  // Both slots empty, or both carrying data: keep the previous slot, flag it.
  {
    KwsI2sPcmState st = NewState(0, 8, 0);
    auto good = MakeHop(1, [](uint32_t) { return 256; });
    std::vector<uint32_t> zeros(2 * kFrames, 0u);
    std::vector<uint32_t> both(2 * kFrames, 0x00000100u);
    KwsI2sHopToPcm16(&st, good.data(), kFrames, out.data(), &stats);
    KwsI2sHopToPcm16(&st, zeros.data(), kFrames, out.data(), &stats);
    CHECK(st.slot == 1u && st.ambiguous_hops == 1u && stats.peak_abs == 0);
    KwsI2sHopToPcm16(&st, both.data(), kFrames, out.data(), &stats);
    CHECK(st.slot == 1u && st.ambiguous_hops == 2u && st.slot_changes == 0u);
  }

  // Saturation instead of wrap-around when the shift adds gain.
  {
    KwsI2sPcmState st = NewState(0, 4, 0);
    auto hi = MakeHop(0, [](uint32_t) { return 8388607; });
    KwsI2sHopToPcm16(&st, hi.data(), kFrames, out.data(), &stats);
    CHECK(out[0] == 32767 && out[319] == 32767);
    auto lo = MakeHop(0, [](uint32_t) { return -8388608; });
    KwsI2sHopToPcm16(&st, lo.data(), kFrames, out.data(), &stats);
    CHECK(out[319] == -32768);
  }

  // Start-up discard: muted hops are zero, then audio passes.
  {
    KwsI2sPcmState st = NewState(0, 8, 2);
    auto words = MakeHop(0, [](uint32_t) { return 25600; });
    KwsI2sHopToPcm16(&st, words.data(), kFrames, out.data(), &stats);
    CHECK(stats.muted == 1u && stats.peak_abs == 0 && out[100] == 0);
    KwsI2sHopToPcm16(&st, words.data(), kFrames, out.data(), &stats);
    CHECK(stats.muted == 1u && st.startup_left == 0u);
    KwsI2sHopToPcm16(&st, words.data(), kFrames, out.data(), &stats);
    CHECK(stats.muted == 0u && out[100] == 100);
  }

  // High-pass: a constant offset decays to ~0 within a second ...
  {
    KwsI2sPcmState st = NewState(60, 8, 0);
    auto dc = MakeHop(0, [](uint32_t) { return 2000000; });
    for (int hop = 0; hop < 50; ++hop) {
      KwsI2sHopToPcm16(&st, dc.data(), kFrames, out.data(), &stats);
    }
    CHECK(stats.peak_abs <= 1 && stats.dc == 0);
  }
  // ... while a 1 kHz tone passes almost unchanged (|H| > 0.99 at 60 Hz cutoff).
  {
    KwsI2sPcmState st = NewState(60, 8, 0);
    const double amp = 1000000.0;
    for (int hop = 0; hop < 10; ++hop) {
      auto tone = MakeHop(0, [&](uint32_t i) {
        const double n = static_cast<double>(hop) * kFrames + i;
        return static_cast<int32_t>(std::lround(amp * std::sin(2.0 * M_PI * 1000.0 * n / 16000.0)));
      });
      KwsI2sHopToPcm16(&st, tone.data(), kFrames, out.data(), &stats);
    }
    const double expect = amp / 256.0;
    CHECK(stats.peak_abs > static_cast<int32_t>(0.98 * expect));
    CHECK(stats.peak_abs <= static_cast<int32_t>(expect) + 1);
    CHECK(stats.dc > -3 && stats.dc < 3);
  }

  std::printf("OK: kws_i2s host check (I2S geometry + PCM16 conditioning)\n");
  return 0;
}
