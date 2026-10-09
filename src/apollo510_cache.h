// SPDX-License-Identifier: Apache-2.0
// DMA cache coherency: invalidate-on-consume. DMA writes do not snoop D-cache.
// Skip DTCM (0x20000000, 512 KiB) — invalidate there has stalled this silicon.
#ifndef APOLLO510_CACHE_H_
#define APOLLO510_CACHE_H_

#include <stddef.h>
#include <stdint.h>

#ifdef KWS_TARGET_APOLLO510
#include "am_hal_cachectrl.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

static inline int Apollo510PtrInDtcm(const void* ptr) {
#ifdef KWS_TARGET_APOLLO510
  const uint32_t a = (uint32_t)(uintptr_t)ptr;
  return a >= 0x20000000u && a < 0x20080000u;
#else
  (void)ptr;
  return 0;
#endif
}

static inline void Apollo510DmaInvalidate(const void* ptr, uint32_t bytes) {
#ifdef KWS_TARGET_APOLLO510
  if (ptr == NULL || bytes == 0 || Apollo510PtrInDtcm(ptr)) return;
  am_hal_cachectrl_range_t range;
  range.ui32StartAddr = (uint32_t)(uintptr_t)ptr;
  range.ui32Size = bytes;
  am_hal_cachectrl_dcache_invalidate(&range, false);
#else
  (void)ptr;
  (void)bytes;
#endif
}

#ifdef __cplusplus
}
#endif
#endif  // APOLLO510_CACHE_H_
