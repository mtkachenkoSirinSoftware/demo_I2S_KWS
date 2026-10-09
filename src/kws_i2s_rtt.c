// SPDX-License-Identifier: Apache-2.0
// Hop PCM on SEGGER RTT channel 1. Called from main after AcquireBlock, not
// from the I2S ISR. Channel 1 is PCM, not a label (labels stay on SWO/ITM).
// The buffer holds 160 ms so a capture never starts with stale audio.
#include "kws_i2s_rtt.h"

#include <stdint.h>

#include "SEGGER_RTT.h"
#include "am_mcu_apollo.h"
#include "config/kws_config.h"
#include "nsx_mem.h"

enum { kKwsRttPcmHops = 8 };
enum { kKwsRttPcmBytes = kKwsRttPcmHops * KWS_AUDIO_BLOCK_SAMPLES * (int)sizeof(int16_t) };

static NSX_MEM_SRAM_BSS uint8_t g_rtt_pcm[kKwsRttPcmBytes];

void KwsI2sRttInit(void) {
  SEGGER_RTT_Init();
  (void)SEGGER_RTT_ConfigUpBuffer(kKwsI2sRttPcmChannel, "PCM", g_rtt_pcm, sizeof(g_rtt_pcm),
                                  SEGGER_RTT_MODE_NO_BLOCK_SKIP);
}

void KwsI2sRttWriteHop(const int16_t* pcm, uint32_t n_samples) {
  if (pcm == NULL || n_samples == 0u) {
    return;
  }
  (void)SEGGER_RTT_Write(kKwsI2sRttPcmChannel, pcm, n_samples * (uint32_t)sizeof(int16_t));
  /* J-Link SWD bypasses the M55 D-cache; CoreMark does the same flush. */
  SCB_CleanDCache();
}
