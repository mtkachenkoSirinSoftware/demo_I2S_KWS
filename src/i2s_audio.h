// SPDX-License-Identifier: Apache-2.0
// I2S microphone on Apollo510 EVB: pins, clock and block geometry.
// Values proven on this board by demo_b/i2s_mic_test (AmbiqSuite HAL).
#ifndef I2S_AUDIO_H_
#define I2S_AUDIO_H_

#include <assert.h>
#include <stdint.h>

#include "audio/i2s_pcm16.h"
#include "config/kws_config.h"

#ifdef __cplusplus
extern "C" {
#endif

#define KWS_I2S_MODULE 0
// Not the BSP default I2S0 pins (5 / 7 / 4): these reach J8 on the EVB.
#define KWS_I2S_CLK_GPIO 47   // I2S0_CLK  -> mic SCK
#define KWS_I2S_WS_GPIO 48    // I2S0_WS   -> mic WS
#define KWS_I2S_SDIN_GPIO 4   // I2S0_SDIN <- mic SD

// SYSPLL 24.576 MHz -> FOUT4 3.072 MHz -> DIV3 -> BCLK 1.024 MHz
// BCLK / (2 slots * 32 bits) = 16 kHz.
#define KWS_I2S_SYSPLL_HZ 24576000u
#define KWS_I2S_BCLK_HZ 1024000u
#define KWS_I2S_SLOT_BITS 32u
#define KWS_I2S_FS_HZ (KWS_I2S_BCLK_HZ / (KWS_I2S_WORDS_PER_FRAME * KWS_I2S_SLOT_BITS))

// One DMA block is one hop: 320 stereo frames = 640 words = 20 ms.
#define KWS_I2S_HOP_FRAMES KWS_AUDIO_BLOCK_SAMPLES
#define KWS_I2S_DMA_WORDS (KWS_I2S_HOP_FRAMES * KWS_I2S_WORDS_PER_FRAME)

#ifndef KWS_I2S_PCM_SHIFT
#define KWS_I2S_PCM_SHIFT 8u
#endif
#ifndef KWS_I2S_HPF_HZ
#define KWS_I2S_HPF_HZ 60u
#endif
// The microphone starts with ~2 s of decaying DC (measured in bring-up).
#ifndef KWS_I2S_STARTUP_HOPS
#define KWS_I2S_STARTUP_HOPS 100u
#endif

static_assert(KWS_I2S_FS_HZ == KWS_SAMPLE_RATE_HZ, "I2S frame rate must equal the KWS sample rate");

// Slot choice, start-up state and slot-change counters of the running source.
const KwsI2sPcmState* I2sAudioPcmState(void);
const KwsI2sHopStats* I2sAudioLastHopStats(void);

// After a failed Init/Start: 1 clkmgr, 2 initialize, 3 power, 4 configure,
// 5 enable, 6 dma_configure, 7 dma_transfer_start; and the HAL status code.
uint32_t I2sAudioFailStep(void);
uint32_t I2sAudioFailStatus(void);

// Stops and restarts the DMA stream after a stall; counts as a dma_fault.
bool I2sAudioRestart(void);

#ifdef __cplusplus
}
#endif
#endif  // I2S_AUDIO_H_
