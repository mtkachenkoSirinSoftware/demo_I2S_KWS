// SPDX-License-Identifier: Apache-2.0
// kws_i2s — always-on KWS from an I2S MEMS microphone through kws_core.
// Labels on SWO; `pred` is the live microphone, not a flash/UART clip.
//
// Always-on loop: one DMA interrupt per 20 ms hop wakes the core, the hop is
// pushed through the PCM ring and the incremental frontend, an inference runs
// every KWS_INFERENCE_STRIDE_FRAMES hops, and the core sleeps until the next
// interrupt.
#include "app/kws_app.h"
#include "audio/audio_source.h"
#include "config/kws_config.h"
#include "i2s_audio.h"
#include "model/model_runner.h"
#include "profiling/kws_profile.h"
#include "profiling/kws_swo_perf.h"

#include "am_mcu_apollo.h"
#include "nsx_core.h"
#include "nsx_system.h"

#if defined(KWS_I2S_RTT_PCM)
#include "kws_i2s_rtt.h"
#endif

namespace {

const nsx_system_config_t kCfg = {
    .perf_mode = NSX_PERF_LOW,
    .enable_cache = true,
    .enable_sram = true,
    .debug = {.transport = NSX_DEBUG_ITM},
    // Full am_bsp_low_power_init(). With the minimal init (as in kws_pdm)
    // am_hal_i2s_enable() fails with AM_HAL_STATUS_INVALID_OPERATION; setting
    // the clock manager's board info and HFRC config by hand did not fix it.
    .skip_bsp_init = false,
    .spot_mgr_profile = true,
};

constexpr uint32_t kHopsPerSecond = 1000u / KWS_AUDIO_BLOCK_MS;
// No hop for this long means the DMA stream stalled (a hop is 20 ms).
constexpr uint32_t kStallMs = 500u;

// CYCCNT cycles spent inside the sleep call since the last report.
uint32_t g_slept_cycles = 0;

const char* LabelName(int lab) {
  return (lab >= 0 && lab < KWS_NUM_CLASSES) ? kws::kLabels[lab] : "-";
}

// Sleeps until the next interrupt unless a hop is already waiting. The check
// and the sleep share one critical section: WFI still wakes on a pending
// interrupt with PRIMASK set, so a hop that lands in between is not slept
// through.
void SleepUntilInterrupt() {
#if defined(KWS_I2S_SLEEP)
  const uint32_t primask = am_hal_interrupt_master_disable();
  if (!AudioSourceBlockReady()) {
    const uint32_t t0 = KwsCycleCount();
    am_hal_sysctrl_sleep(AM_HAL_SYSCTRL_SLEEP_NORMAL);
    g_slept_cycles += KwsCycleCount() - t0;
  }
  am_hal_interrupt_master_set(primask);
#endif
}

bool RunInference(kws::KwsApp& app, uint32_t now_ms) {
  const bool fired = app.RunInference(now_ms);
#if defined(KWS_I2S_INFER_LINES)
  nsx_printf("inf pred=%s score=%.3f\n", LabelName(app.last_label()), app.last_score());
#endif
  if (fired) {
    nsx_printf("event label=%s score=%.3f\n", LabelName(app.last_event().label),
               app.last_event().score);
  }
  return fired;
}

void PrintSecond(const kws::KwsApp& app, bool fired, uint32_t awake_permille) {
  const AudioSourceStatus* st = AudioSourceGetStatus();
  const KwsI2sPcmState* pcm = I2sAudioPcmState();
  const kws::Telemetry& t = app.telemetry();
  nsx_printf("Frame %lu  peak=%d dc=%d overruns=%u dma_faults=%u slot=%u slot_changes=%u "
             "muted=%u awake=%u.%u\n",
             (unsigned long)st->blocks_received, (int)st->pcm_peak_abs, (int)st->pcm_dc,
             (unsigned)st->overruns, (unsigned)st->dma_faults, (unsigned)pcm->slot,
             (unsigned)pcm->slot_changes, (unsigned)I2sAudioLastHopStats()->muted,
             (unsigned)(awake_permille / 10u), (unsigned)(awake_permille % 10u));
  nsx_printf("stage frontend_cycles=%u mfcc_cycles=%u quantize_cycles=%u invoke_cycles=%u\n",
             (unsigned)t.frontend_cycles_last, (unsigned)t.mfcc_cycles_last,
             (unsigned)t.quantize_cycles_last, (unsigned)t.inference_cycles_last);
  nsx_printf("live pred=%s score=%.3f fired=%d invoke_cycles=%u invoke_us=%u\n",
             LabelName(app.last_label()), app.last_score(), fired ? 1 : 0,
             (unsigned)t.inference_cycles_last,
             (unsigned)kws::CyclesToMicroseconds(t.inference_cycles_last));
}

}  // namespace

int main(void) {
  NSX_TRY(nsx_system_init(&kCfg), "System init failed\n");
  nsx_itm_printf_enable();
  KwsProfileInit();

  kws::KwsApp app;
  if (!app.Init()) {
    nsx_printf("ERROR: KwsApp::Init failed\n");
    for (;;) {
    }
  }
  if (!AudioSourceInit() || !AudioSourceStart()) {
    nsx_printf("ERROR: I2S source failed step=%u hal_status=%u\n", (unsigned)I2sAudioFailStep(),
               (unsigned)I2sAudioFailStatus());
    for (;;) {
    }
  }

#if defined(KWS_I2S_RTT_PCM)
  KwsI2sRttInit();
#endif

  nsx_printf("kws_i2s runtime=%s arena_used=%u\n", kws::ModelRunner::runtime_name(),
             static_cast<unsigned>(kws::ModelRunner::arena_used_bytes()));
  KwsPrintPerfBanner(kCfg.perf_mode);
  nsx_printf("i2s backend=hal fs=%u bclk=%u slots=%ux%u bits=%u shift=%u hpf=%u startup_hops=%u\n",
             (unsigned)KWS_I2S_FS_HZ, (unsigned)KWS_I2S_BCLK_HZ,
             (unsigned)KWS_I2S_WORDS_PER_FRAME, (unsigned)KWS_I2S_SLOT_BITS,
             (unsigned)KWS_I2S_SAMPLE_BITS, (unsigned)KWS_I2S_PCM_SHIFT, (unsigned)KWS_I2S_HPF_HZ,
             (unsigned)KWS_I2S_STARTUP_HOPS);
  nsx_printf("gpio clk=%u ws=%u sdin=%u\n", (unsigned)KWS_I2S_CLK_GPIO, (unsigned)KWS_I2S_WS_GPIO,
             (unsigned)KWS_I2S_SDIN_GPIO);
  nsx_printf("model sha256=%s\n", KWS_MODEL_SHA256);
  nsx_printf("always_on sleep=%d infer_lines=%d rtt_pcm=%d stride_ms=%u\n",
#if defined(KWS_I2S_SLEEP)
             1,
#else
             0,
#endif
#if defined(KWS_I2S_INFER_LINES)
             1,
#else
             0,
#endif
#if defined(KWS_I2S_RTT_PCM)
             1,
#else
             0,
#endif
             (unsigned)(KWS_INFERENCE_STRIDE_FRAMES * KWS_AUDIO_BLOCK_MS));
  nsx_printf("pred is the live I2S microphone, not a flash/UART clip\n");

  const uint32_t cpu_hz = KwsCpuHz();
  const uint32_t stall_cycles = (cpu_hz / 1000u) * kStallMs;
  uint32_t now_ms = 0;
  uint32_t hops_this_second = 0;
  bool fired_this_second = false;
  uint32_t window_start = KwsCycleCount();
  uint32_t last_hop_cycles = window_start;

  for (;;) {
    unsigned drained = 0;
    while (AudioSourceBlockReady() && drained < 8u) {
      uint32_t count = 0;
      const AudioSample* block = AudioSourceAcquireBlock(&count);
      if (block != nullptr) {
#if defined(KWS_I2S_RTT_PCM)
        KwsI2sRttWriteHop(block, count);
#endif
        app.OnAudioBlock(block, count, now_ms);
      }
      AudioSourceReleaseBlock();
      now_ms += KWS_AUDIO_BLOCK_MS;
      last_hop_cycles = KwsCycleCount();
      ++hops_this_second;
      ++drained;
      if (app.InferenceDue()) {
        fired_this_second |= RunInference(app, now_ms);
      }

      if (hops_this_second >= kHopsPerSecond) {
        // One second by the microphone's clock. Awake share = cycles outside
        // the sleep call / cycles in one second at the reported cpu_hz.
        const uint32_t now_cycles = KwsCycleCount();
        const uint32_t active = (now_cycles - window_start) - g_slept_cycles;
        const uint32_t awake_permille =
            (cpu_hz != 0u) ? (uint32_t)(((uint64_t)active * 1000ull) / cpu_hz) : 0u;
        PrintSecond(app, fired_this_second, awake_permille);
        window_start = now_cycles;  // printing counts as awake time
        g_slept_cycles = 0;
        hops_this_second = 0;
        fired_this_second = false;
      }
    }

    // Stall watchdog. It relies on CYCCNT advancing, so it cannot fire while
    // the core sleeps with no interrupt at all.
    if (stall_cycles != 0u && (KwsCycleCount() - last_hop_cycles) > stall_cycles) {
      const bool ok = I2sAudioRestart();
      nsx_printf("ERROR: i2s stalled, restart %s\n", ok ? "ok" : "failed");
      last_hop_cycles = KwsCycleCount();
    }

    SleepUntilInterrupt();
  }
}
