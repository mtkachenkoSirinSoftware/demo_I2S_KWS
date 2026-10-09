// SPDX-License-Identifier: Apache-2.0
// Cycle-counter seam.
//
// This exists so the application can time its own stages *on the target*
// without dsp/, model/ or app/ acquiring a hardware header. The counter is a
// free function resolved at link time: platform/apollo510_profile.c on the EVB,
// profiling/kws_profile_host.cc on the host.
//
// What this is NOT
// ----------------
// On the host, KwsCycleCount() returns 0 and every scope measures zero. A
// host run must not look like a latency, cycle, or stall figure. A non-zero
// cycle count from this header means an Apollo510 produced it.
//
// How it relates to heliaPROFILER
// -------------------------------
// `hpx` measures the *model* — per-layer PMU counters inside firmware it
// generates itself, with no audio pipeline attached. This header measures the
// *application*: frontend cost, quantisation cost, and the inference call as
// the app actually issues it. They answer different questions and neither
// substitutes for the other.
#ifndef KWS_PROFILE_H_
#define KWS_PROFILE_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Free-running 32-bit cycle counter, or 0 when the platform has none.
// On Apollo510 this is DWT->CYCCNT, which wraps every ~45 s at 96 MHz; every
// consumer below takes an unsigned difference, which is wrap-correct for any
// interval shorter than one wrap.
uint32_t KwsCycleCount(void);

// CPU frequency the cycle counter runs at, or 0 when unknown. Reported rather
// than assumed so a cycles->microseconds conversion can never be done against a
// hard-coded clock that the board was not actually running at.
uint32_t KwsCpuHz(void);

// Called once at start-up to enable the counter. No-op where there is none.
void KwsProfileInit(void);

#ifdef __cplusplus
}  // extern "C"

namespace kws {

// Scoped stage timer. Records elapsed cycles into `*out` on destruction.
class CycleScope {
 public:
  explicit CycleScope(uint32_t* out) : out_(out), start_(KwsCycleCount()) {}
  ~CycleScope() {
    if (out_ != nullptr) *out_ = KwsCycleCount() - start_;
  }
  CycleScope(const CycleScope&) = delete;
  CycleScope& operator=(const CycleScope&) = delete;

 private:
  uint32_t* out_;
  uint32_t start_;
};

// Cycles -> microseconds, using the clock the platform reports. Returns 0 when
// the clock is unknown, rather than inventing a divisor.
inline uint32_t CyclesToMicroseconds(uint32_t cycles) {
  const uint32_t hz = KwsCpuHz();
  if (hz == 0) return 0;
  return (uint32_t)(((uint64_t)cycles * 1000000ull) / hz);
}

}  // namespace kws
#endif  // __cplusplus
#endif  // KWS_PROFILE_H_
