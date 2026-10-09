// SPDX-License-Identifier: Apache-2.0
// Apollo510 DWT cycle counter for kws_core CycleScope. Not heliaPROFILER.
#include "profiling/kws_profile.h"

#include "am_mcu_apollo.h"

void KwsProfileInit(void) {
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

uint32_t KwsCycleCount(void) { return DWT->CYCCNT; }

uint32_t KwsCpuHz(void) { return (uint32_t)SystemCoreClock; }
