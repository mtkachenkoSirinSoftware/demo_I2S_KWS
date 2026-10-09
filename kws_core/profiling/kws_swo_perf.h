// SPDX-License-Identifier: Apache-2.0
// SWO helpers for firmware mains (needs nsx_printf). Not a host measurement.
#ifndef KWS_SWO_PERF_H_
#define KWS_SWO_PERF_H_

#include "nsx_core.h"
#include "nsx_system.h"
#include "profiling/kws_profile.h"

#ifdef __cplusplus

inline const char* KwsPerfModeName(nsx_perf_mode_e mode) {
  switch (mode) {
    case NSX_PERF_LOW:
      return "LOW";
    case NSX_PERF_MEDIUM:
      return "MEDIUM";
    case NSX_PERF_HIGH:
      return "HIGH";
    default:
      return "?";
  }
}

inline void KwsPrintPerfBanner(nsx_perf_mode_e mode) {
  nsx_printf("perf_mode=%s cpu_hz=%u\n", KwsPerfModeName(mode),
             static_cast<unsigned>(KwsCpuHz()));
}

// Prints DWT cycles and, only when KwsCpuHz() is non-zero, microseconds.
// Divisor is the printed cpu_hz (NSX_PERF_LOW is 96 MHz on Apollo510), not
// a guessed 192/250 MHz and not hpx profile.
inline void KwsPrintInvokeTail(uint32_t cycles) {
  const uint32_t us = kws::CyclesToMicroseconds(cycles);
  if (us != 0) {
    nsx_printf(" invoke_cycles=%u invoke_us=%u\n", static_cast<unsigned>(cycles),
               static_cast<unsigned>(us));
    nsx_printf("  invoke_us = CYCCNT / cpu_hz (see perf_mode=); not hpx profile\n");
  } else {
    nsx_printf(" invoke_cycles=%u\n", static_cast<unsigned>(cycles));
    nsx_printf("  invoke_cycles is DWT CYCCNT of this binary's Invoke; "
               "invoke_us omitted (cpu_hz unknown)\n");
  }
}

#endif  // __cplusplus
#endif  // KWS_SWO_PERF_H_
