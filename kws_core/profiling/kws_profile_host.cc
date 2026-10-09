// SPDX-License-Identifier: Apache-2.0
// Host implementation of the cycle-counter seam: there is none.
//
// Returning 0 rather than a std::chrono reading is the whole point — see the
// header. A host build must not be able to emit a number that looks like an
// Apollo510 cycle count.
#include "profiling/kws_profile.h"

extern "C" uint32_t KwsCycleCount(void) { return 0; }
extern "C" uint32_t KwsCpuHz(void) { return 0; }
extern "C" void KwsProfileInit(void) {}
