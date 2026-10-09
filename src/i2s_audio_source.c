// SPDX-License-Identifier: Apache-2.0
// I2S microphone AudioSource on the AmbiqSuite HAL (nsx-audio has no I2S).
// I2S0 is the controller and only receives; ping-pong DMA delivers one
// 20 ms hop per interrupt. The ISR only flags the block: slot pick,
// high-pass and the 24->16 bit conversion run in the main loop.
#include "audio/audio_source.h"

#include <stdbool.h>
#include <string.h>

#include "apollo510_cache.h"
#include "i2s_audio.h"

#include "am_mcu_apollo.h"
#include "am_util.h"
#include "nsx_mem.h"

static NSX_MEM_SRAM_BSS uint32_t __attribute__((aligned(32)))
    g_dma[2 * KWS_I2S_DMA_WORDS];
static AudioSample g_hop[KWS_I2S_HOP_FRAMES];
static AudioSourceStatus g_status;
static KwsI2sPcmState g_pcm;
static KwsI2sHopStats g_hop_stats;
static void* g_i2s;
static volatile uint8_t g_ready;
static uint32_t g_fail_step;
static uint32_t g_fail_status;

// Records which HAL step failed so main can print it; returns true on success.
static bool Step(uint32_t step, uint32_t status) {
  if (status == AM_HAL_STATUS_SUCCESS) return true;
  g_fail_step = step;
  g_fail_status = status;
  return false;
}

static am_hal_i2s_io_signal_t g_io_cfg = {
    .sFsyncPulseCfg = {.eFsyncPulseType = AM_HAL_I2S_FSYNC_PULSE_ONE_SUBFRAME},
    .eFyncCpol = AM_HAL_I2S_IO_FSYNC_CPOL_LOW,
    .eTxCpol = AM_HAL_I2S_IO_TX_CPOL_FALLING,
    .eRxCpol = AM_HAL_I2S_IO_RX_CPOL_RISING,
};

static am_hal_i2s_data_format_t g_data_cfg = {
    .ePhase = AM_HAL_I2S_DATA_PHASE_SINGLE,
    .eChannelLenPhase1 = AM_HAL_I2S_FRAME_WDLEN_32BITS,
    .ui32ChannelNumbersPhase1 = KWS_I2S_WORDS_PER_FRAME,
    .eDataDelay = 0x1,
    .eSampleLenPhase1 = AM_HAL_I2S_SAMPLE_LENGTH_24BITS,
    .eDataJust = AM_HAL_I2S_DATA_JUSTIFIED_LEFT,
};

static am_hal_i2s_config_t g_i2s_cfg = {
    .eMode = AM_HAL_I2S_IO_MODE_MASTER,
    .eXfer = AM_HAL_I2S_XFER_RX,
    .eClock = eAM_HAL_I2S_CLKSEL_PLL_FOUT4,
    .eDiv3 = 1,
    .eASRC = 0,
    .eData = &g_data_cfg,
    .eIO = &g_io_cfg,
};

static am_hal_i2s_transfer_t g_transfer = {
    .ui32RxTotalCount = KWS_I2S_DMA_WORDS,
    .ui32RxTargetAddr = (uint32_t)(&g_dma[0]),
    .ui32RxTargetAddrReverse = (uint32_t)(&g_dma[KWS_I2S_DMA_WORDS]),
};

void am_dspi2s0_isr(void) {
  uint32_t status;
  am_hal_i2s_interrupt_status_get(g_i2s, &status, true);
  am_hal_i2s_interrupt_clear(g_i2s, status);
  // Switches the ping-pong buffer.
  am_hal_i2s_interrupt_service(g_i2s, status, &g_i2s_cfg);
  if (status & AM_HAL_I2S_INT_RXDMACPL) {
    if (g_ready) {
      ++g_status.overruns;
    }
    ++g_status.blocks_received;
    g_ready = 1;
  }
}

static void ConfigPin(uint32_t pin, uint32_t func, am_hal_gpio_drivestrength_e drive) {
  am_hal_gpio_pincfg_t cfg = {0};
  cfg.GP.cfg_b.uFuncSel = func;
  cfg.GP.cfg_b.eDriveStrength = drive;
  am_hal_gpio_pinconfig(pin, cfg);
}

bool AudioSourceInit(void) {
  for (int i = 0; i < 2; ++i) g_status.state[i] = kBufferFree;
  g_status.blocks_received = 0;
  g_status.blocks_processed = 0;
  g_status.overruns = 0;
  g_status.uart_errors = 0;
  g_status.dma_faults = 0;
  g_status.pcm_peak_abs = 0;
  g_status.pcm_dc = 0;
  g_ready = 0;
  memset(&g_hop_stats, 0, sizeof(g_hop_stats));
  KwsI2sPcmInit(&g_pcm, KWS_I2S_FS_HZ, KWS_I2S_HPF_HZ, KWS_I2S_PCM_SHIFT, KWS_I2S_STARTUP_HOPS);

  if (!Step(1, am_hal_clkmgr_clock_config(AM_HAL_CLKMGR_CLK_ID_SYSPLL, KWS_I2S_SYSPLL_HZ, NULL))) {
    return false;
  }
  ConfigPin(KWS_I2S_CLK_GPIO, AM_HAL_PIN_47_I2S0_CLK, AM_HAL_GPIO_PIN_DRIVESTRENGTH_1P0X);
  ConfigPin(KWS_I2S_WS_GPIO, AM_HAL_PIN_48_I2S0_WS, AM_HAL_GPIO_PIN_DRIVESTRENGTH_1P0X);
  ConfigPin(KWS_I2S_SDIN_GPIO, AM_HAL_PIN_4_I2S0_SDIN, AM_HAL_GPIO_PIN_DRIVESTRENGTH_0P1X);

  if (!Step(2, am_hal_i2s_initialize(KWS_I2S_MODULE, &g_i2s))) return false;
  if (!Step(3, am_hal_i2s_power_control(g_i2s, AM_HAL_I2S_POWER_ON, false))) return false;
  if (!Step(4, am_hal_i2s_configure(g_i2s, &g_i2s_cfg))) return false;
  return Step(5, am_hal_i2s_enable(g_i2s));
}

bool AudioSourceStart(void) {
  if (!Step(6, am_hal_i2s_dma_configure(g_i2s, &g_i2s_cfg, &g_transfer))) return false;
  NVIC_SetPriority(I2S0_IRQn, AM_IRQ_PRIORITY_DEFAULT);
  NVIC_ClearPendingIRQ(I2S0_IRQn);
  NVIC_EnableIRQ(I2S0_IRQn);
  return Step(7, am_hal_i2s_dma_transfer_start(g_i2s, &g_i2s_cfg));
}

bool AudioSourceStop(void) {
  NVIC_DisableIRQ(I2S0_IRQn);
  return am_hal_i2s_dma_transfer_complete(g_i2s) == AM_HAL_STATUS_SUCCESS;
}

bool I2sAudioRestart(void) {
  ++g_status.dma_faults;
  (void)AudioSourceStop();
  (void)am_hal_i2s_disable(g_i2s);
  g_ready = 0;
  if (am_hal_i2s_enable(g_i2s) != AM_HAL_STATUS_SUCCESS) return false;
  return AudioSourceStart();
}

bool AudioSourceBlockReady(void) { return g_ready != 0; }

const AudioSample* AudioSourceAcquireBlock(uint32_t* out_samples) {
  if (!g_ready) return NULL;
  // Claim the block and read "last completed buffer" as one step: the flag is
  // cleared here, not on release, so a hop that completes while this one is
  // being processed is kept instead of being counted as an overrun.
  const uint32_t primask = am_hal_interrupt_master_disable();
  g_ready = 0;
  const uint32_t* words = (const uint32_t*)am_hal_i2s_dma_get_buffer(g_i2s, AM_HAL_I2S_XFER_RX);
  am_hal_interrupt_master_set(primask);
  if (words == NULL) return NULL;
  Apollo510DmaInvalidate(words, sizeof(uint32_t) * KWS_I2S_DMA_WORDS);
  g_status.state[g_status.blocks_processed & 1u] = kBufferCpuProcessing;
  KwsI2sHopToPcm16(&g_pcm, words, KWS_I2S_HOP_FRAMES, g_hop, &g_hop_stats);
  g_status.pcm_peak_abs = g_hop_stats.peak_abs;
  g_status.pcm_dc = g_hop_stats.dc;
  if (out_samples != NULL) *out_samples = KWS_I2S_HOP_FRAMES;
  return g_hop;
}

void AudioSourceReleaseBlock(void) {
  g_status.state[g_status.blocks_processed & 1u] = kBufferFree;
  ++g_status.blocks_processed;
}

const AudioSourceStatus* AudioSourceGetStatus(void) { return &g_status; }
const KwsI2sPcmState* I2sAudioPcmState(void) { return &g_pcm; }
uint32_t I2sAudioFailStep(void) { return g_fail_step; }
uint32_t I2sAudioFailStatus(void) { return g_fail_status; }
const KwsI2sHopStats* I2sAudioLastHopStats(void) { return &g_hop_stats; }

bool AudioSourceTakeStreamBegin(void) { return false; }
bool AudioSourceTakeStreamEnd(void) { return false; }
bool AudioSourceTakeReset(void) { return false; }
