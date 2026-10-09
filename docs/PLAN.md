# demo_I2S — always-on KWS from an I2S microphone on Apollo510 EVB

Original overview, written before implementation. Current state and run
instructions: [README.md](../README.md).

> Superseded where they differ by [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md),
> which records the decisions taken since: `kws_core` is **copied** into this
> project (not referenced), the nsx build is confirmed viable, and current
> measurement is out of scope.

**Goal.** A firmware image that runs the shipping INT8 KWS model on live audio
from an I2S MEMS microphone, and an English browser dashboard that shows, in
real time, whether a spoken keyword was recognized.

**Why a new project.** No existing example in this repository combines KWS with
an I2S microphone (survey in §1). The PDM path (`kws_pdm`) delivered no live
audio on this board; the I2S microphone does (§2).

---

## 1. What already exists

| Location | What it is | Reused here |
|---|---|---|
| `ambiq_kws_examples/kws_core/` | KWS library: streaming MFCC, model runner, recognizer, `AudioSource` contract | Yes, unchanged |
| `ambiq_kws_examples/kws_pdm/` | KWS on a PDM microphone (nsx app), SWO labels, optional RTT PCM | Structure, `main.cc`, RTT dump |
| `ambiq_kws_examples/kws_pdm/host/kws_dashboard.{py,html}` | Live dashboard (Russian UI) reading SWO | Base for the English dashboard |
| `ambiq_kws_examples/kws_clip`, `kws_uart`; `demo_b/`; `apollo510_kws_demo/` | KWS from a flash clip / UART / USB | No (other audio origins) |
| `demo_b/i2s_mic_test/` | I2S microphone bring-up firmware + RTT WAV capture, no KWS | I2S config, pins, decoder findings |
| AmbiqSuite 5.2.0 `examples/audio/i2s_loopback` | I2S0↔I2S1 loopback, no microphone, no KWS | HAL call sequence |
| `nsx-audio` module | PDM only (`NSX_AUDIO_SOURCE_PDM`) | No |
| `third_party/neuralspotx/examples/kws_infer` | Dummy-input inference | No |
| `third_party/tiny` | I2S **output** sink on STM32, no microphone capture for Apollo | No |

## 2. What the bring-up established (measured on this EVB, 2026-10-08)

Source: `demo_b/i2s_mic_test`, firmware `i2s_mic_rtt`, captures in its `capture/`.

- **Wiring that works:** SCK → GPIO 47 (J8-16), WS → GPIO 48 (J8-18),
  SD → GPIO 4 (J8-9), L/R → GND, VDD → VDD_MCU 1.8 V. These are valid
  `I2S0_CLK` / `I2S0_WS` / `I2S0_SDIN` functions but **not** the BSP default
  I2S0 pins (5 / 7 / 4), so the pins are configured directly.
- **Format that works:** I2S0 as controller, RX only, standard I2S, two 32-bit
  slots, 24-bit samples in bits [23:0], 16 kHz
  (SYSPLL 24.576 MHz → FOUT4 3.072 MHz → DIV3 → BCLK 1.024 MHz).
- **Delivery:** 125 DMA buffers/s of 256 words, none missed.
- **Level:** room background about −54 dBFS RMS; the user confirmed speech is
  audible in a capture.
- **Start-up transient:** about 2 s of large decaying DC after reset.
- **Slot position is not fixed:** the microphone sample was seen in the first
  word of the stereo pair in one capture and in the second word in others. The
  empty slot reads exactly zero. Cause not established (§7).

## 3. Design

### 3.1 Layout

```
demo_I2S/
  PLAN.md                 this file
  README.md               wiring, build, run, expected output        (M4)
  CMakeLists.txt, nsx.yml nsx app, same module pins as kws_pdm       (M0)
  src/main.cc             init, drain hops, SWO lines                 (M2)
  src/i2s_audio_source.c  AudioSource over HAL I2S                    (M1)
  src/i2s_audio.h         pins, clock, format constants               (M1)
  audio/i2s_pcm16.h       pure C: slot pick, 24→16, DC blocker        (M1)
  src/kws_i2s_rtt.c       optional RTT ch1 hop PCM                    (M1)
  host/kws_dashboard.py   SWO → SSE server                            (M3)
  host/kws_dashboard.html English dashboard                           (M3)
  host/rtt_pcm_dump.py    RTT ch1 → WAV                               (M1)
  tests/                  host tests for i2s_pcm16.h and the parser   (M1, M3)
  model/generated/        embedded model (generated, git-ignored)     (M0)
```

`kws_core` is **referenced, not copied**: `KWS_CORE_ROOT` defaults to
`../ambiq_kws_examples/kws_core`, and the `ambiq_kws_examples` Git SHA is
printed in the README's provenance block. Copying would create a second
recognizer to keep in sync. (Decision to confirm — §8.)

### 3.2 Firmware

**Audio source (`src/i2s_audio_source.c`)** implements the existing
`audio/audio_source.h` contract, so `kws_core` and `main.cc` need no change in
how they consume audio.

| Item | Choice | Reason |
|---|---|---|
| Driver | AmbiqSuite HAL `am_hal_i2s_*` directly | `nsx-audio` has no I2S source |
| Clock | SYSPLL 24.576 MHz, `PLL_FOUT4`, `eDiv3 = 1` | Exact 16 kHz; proven in bring-up |
| DMA block | 640 words = 320 stereo frames = one 20 ms hop | One block → one MFCC frame, as `kws_core` requires |
| Buffering | HAL ping-pong, two blocks | Same as `i2s_loopback` |
| Slot handling | Per hop, take the slot that is non-zero | Position is not fixed (§2) |
| 24 → 16 bit | Arithmetic shift, default `>> 8`, build option for gain | `kws_core` works on int16 |
| DC removal | One-pole high-pass before the int16 cast | `kws_core` normalizes by the window's signed max, so a DC offset shifts every feature |
| Start-up | Discard hops for a configurable time (default 2 s) | Observed transient (§2) |
| Status | `blocks_received`, `overruns`, `dma_faults`, `pcm_peak_abs`, `pcm_dc` | Existing `AudioSourceStatus` |

Slot pick, bit conversion and the DC blocker live in `audio/i2s_pcm16.h` as
pure C with no HAL includes, so they are unit-tested on the host (the pattern
`kws_pdm/audio/pdm_pcm16.h` already uses).

**SWO output (`src/main.cc`)** keeps the line shapes of `kws_pdm`, so one
parser serves both:

```
kws_i2s runtime=helia-rt arena_used=<N>
perf_mode=LOW cpu_hz=96000000
i2s backend=hal fs=16000 bclk=1024000 slots=2x32 bits=24 shift=8
gpio clk=47 ws=48 sdin=4
model sha256=<64 hex>
Frame <N>  peak=<n> dc=<n> overruns=<n> dma_faults=<n> slot=<0|1>
live pred=<label> score=<s> fired=<0|1> invoke_cycles=<DWT> invoke_us=<us>
event label=<label> score=<s>
```

One addition: an optional compact line per inference (10/s),
`inf pred=<label> score=<s>`, behind `-DKWS_I2S_INFER_LINES=ON`. `kws_pdm`
prints the argmax once per second, which makes the dashboard timeline coarse.
SWO at 1 MHz has room for it; this is verified in M2, not assumed.

**Optional RTT channel 1** (`-DKWS_I2S_RTT_PCM=ON`): the int16 hop the model
actually receives, to check by ear and by level what KWS is fed. The up-buffer
is kept small (≤ 250 ms) so a capture never starts with stale audio — the
bring-up tool had a 2 s stale prefix.

### 3.3 Dashboard (English)

Same architecture as the existing one: `kws_dashboard.py` (stdlib only) spawns
`JLinkSWOViewerCL`, parses lines, and serves `kws_dashboard.html` on
`127.0.0.1` with Server-Sent Events. It stays a local page because a hosted
page cannot reach the J-Link.

Changes against the `kws_pdm` dashboard:

| Area | Change |
|---|---|
| Language | All UI text in English |
| Verdict card | "Recognized: yes", "Fired on a different word", "Not recognized", "No live signal", "Waiting for the board" |
| Word check | Unchanged flow: pick one of 10 keywords, start a 4 s attempt, say it; tally of hits / wrong word / misses |
| Timeline | Per-inference scores when `inf` lines are present; falls back to the 1 Hz `live pred=` |
| Header chips | I2S clock, slot, pins instead of PDM backend |
| Signal health | `peak=0`, frozen `peak`/`dc`, or a changing `slot=` raise a visible warning |
| Parser | Accepts both `kws_i2s` and `kws_pdm` banners; unit-tested against captured logs |
| Replay | `--replay` of a recorded SWO log, used for tests and for demos without a board |

Visual design follows the existing "Apollo510 KWS Profile" page (same tokens,
light and dark).

### 3.4 What the demo may and may not claim

- A live-microphone `pred` is **not** an accuracy figure. The hit / miss tally
  is a bench check and is never written to `results/`.
- `invoke_cycles` / `invoke_us` are DWT counts printed by this binary on the
  EVB. They are reported as "DWT, this firmware", never compared with
  heliaPROFILER or host numbers.
- A live `pred` is never compared with the flash-clip or UART result: the PCM
  is different (`EXAMPLES_CONTRACT.md`).
- Latency, energy and RAM beyond what this binary prints stay "deferred —
  needs a dedicated Apollo510 measurement".

## 4. Milestones

Each milestone ends with something observable; the next one does not start
until it passes.

| # | Deliverable | Done when |
|---|---|---|
| **M0** Scaffold | nsx app that links `kws_core` + model and runs the recognizer on silence | Builds in the devcontainer; banner and `model sha256=` appear on SWO; `libam_hal` in the nsx SDK is confirmed to export `am_hal_i2s_*` |
| **M1** I2S audio source | `i2s_audio_source.c`, `i2s_pcm16.h`, RTT hop dump, host tests | Host tests pass; `Frame` lines show 50 hops/s with `overruns=0`; a 10 s RTT capture of speech is intelligible, with DC under ±50 counts after the start-up discard |
| **M2** Live KWS | `main.cc` wired to the I2S source, `inf` lines | Each of the 10 keywords fires at least once when spoken at about 30 cm in a quiet room; no SWO line corruption over 5 min with `inf` lines on |
| **M3** Dashboard | English dashboard + parser tests + replay fixture | Parser tests pass on a recorded log; with the board attached, a spoken keyword turns the verdict card green within 1 s of the `event` line; unplugging SD raises the signal warning |
| **M4** Demo package | `README.md` (wiring photo/table, build, flash, run, troubleshooting), one-command launcher, recorded SWO log for replay | A second person reproduces the demo from the README alone |

Process: each milestone goes through the repository's agents pipeline
(Coder ≠ Reviewer ≠ sign-off, `agents/README.md`).

## 5. Build and run (target shape)

```bash
# devcontainer (nsx lives there)
cd demo_I2S
nsx lock --app-dir . && nsx configure --app-dir . && nsx build --app-dir .
nsx flash --app-dir .

# host
python3 host/kws_dashboard.py --reset      # http://127.0.0.1:8765/
```

## 6. Test plan

| Level | What | Where |
|---|---|---|
| Host unit | Slot pick (sample in word 0, in word 1, both zero), 24→16 shift and saturation, DC blocker settling | `tests/`, no board |
| Host unit | SWO parser on recorded `kws_i2s` and `kws_pdm` logs, including a dropped first byte and merged lines (both seen on this board) | `tests/` |
| Board | Hop rate, overruns, DC after discard | SWO `Frame` lines |
| Board | What the model hears | RTT hop capture → WAV |
| Board | Keyword pass over all 10 words, repeated 5× each, tally recorded as a bench note | Dashboard word check |

## 7. Risks and open technical questions

| Risk | Why it matters | Plan |
|---|---|---|
| Slot position changes | A fixed "left" read gives silence (this happened in bring-up) | Detect per hop; print `slot=`; investigate in M1 whether the swap only follows a debugger halt (FIFO slip) or also occurs free-running |
| I2S HAL missing from the nsx prebuilt `libam_hal` | Would block the nsx build | Checked first in M0; fallback is to build as an AmbiqSuite `make` project like `i2s_mic_test` |
| Microphone gain | −54 dBFS background at `>> 8`; speech may use few of the 16 bits | Measure speech peak in M1; pick the shift from data |
| GPIO 47 / 48 are also IOM5 and UART2 RX in the BSP | A later feature using those peripherals would conflict | Nothing in this demo uses them; documented in the README |
| SWO throughput with `inf` lines | Dropped characters corrupt the parser | Measured in M2; option stays OFF if it corrupts |
| Model trained on GSC, not on this microphone | Live recognition rate may be well below the dataset accuracy | Reported as observed; not tuned against the test split |

## 8. Decisions needed from the owner

1. **`kws_core`: reference or copy?** Plan assumes a reference to
   `../ambiq_kws_examples/kws_core`. A copy makes `demo_I2S` self-contained
   (like `demo_a` / `demo_b`) at the cost of a second copy to maintain.
2. **Microphone part number.** Needed for the README and to check its minimum
   clock and supply range against 1.024 MHz BCLK and 1.8 V.
3. **Home of the project.** Plan keeps `demo_I2S` at the repository top level
   as asked. An alternative is `ambiq_kws_examples/kws_i2s`, next to its PDM
   sibling, which would make the `kws_core` path trivial.
