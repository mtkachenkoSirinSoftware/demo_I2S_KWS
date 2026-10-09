# demo_I2S — always-on KWS from an I2S microphone

Keyword spotting on an Apollo510 EVB, listening continuously to an I2S MEMS
microphone, with a live browser dashboard. Firmware name: `kws_i2s`.

The board classifies the last second of audio every 100 ms and sleeps between
20 ms audio blocks. Recognized words go out on SWO; the dashboard reads them
and shows whether the word you just said was recognized.

## What you need

**Hardware**

| Item | Notes |
|---|---|
| Ambiq Apollo510 EVB | on-board J-Link; one USB cable to the PC |
| I2S MEMS microphone breakout | 24-bit, 1.6–3.6 V supply (the listing matches an INMP441; the marking on the part was not checked) |
| 6 jumper wires | see the wiring table below |

**Software to download**

| What | Tested version | Where |
|---|---|---|
| Arm GNU Toolchain, `arm-none-eabi` | 15.3.Rel1 | <https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads> |
| SEGGER J-Link Software and Documentation Pack | V9.60 | <https://www.segger.com/downloads/jlink/> |
| Python | 3.14 and 3.11 (needs ≥ 3.10) | your OS |
| CMake, Ninja, git | CMake 4.4.3, Ninja 1.13.2 | your OS |
| `neuralspotx` (the `nsx` build tool) | 0.7.17 | `pip`, see below |

Everything else — the Ambiq SDK, the heliaRT runtime and the CMSIS-NN kernels —
is downloaded by `nsx` at the exact revisions pinned in `nsx.lock` (about
290 MB into `modules/`). The model, the KWS library and the dashboard are in
this repository.

Tested on Linux x86-64. Windows and macOS were not tried.

## Wiring

| Microphone pin | EVB | GPIO |
|---|---|---|
| SCK | J8-16 | 47 (`I2S0_CLK`) |
| WS | J8-18 | 48 (`I2S0_WS`) |
| SD | J8-9 | 4 (`I2S0_SDIN`) |
| L/R | J9-7 | GND (left slot) |
| VDD | J9-12 | VDD_MCU, 1.8 V |
| GND | J8-19 | GND |

GPIO 47 / 48 are also IOM5 and UART2 RX in the BSP — do not use those
peripherals with this demo.

## Build and run from a fresh clone

Run these in this directory (the one with `nsx.yml`).

```bash
# 1. The nsx build tool, in a virtual environment
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 2. Put the toolchain and J-Link on PATH (adjust to where you unpacked them)
export PATH="$HOME/arm-gnu-toolchain-15.3.rel1-x86_64-arm-none-eabi/bin:/opt/SEGGER/JLink:$PATH"

# 3. Check the tools
make doctor

# 4. Build. The first run downloads the pinned modules (~290 MB, needs network)
make build            # -> build/apollo510_evb/kws_i2s.bin

# 5. Connect the EVB over USB, flash, open the dashboard
make flash
make dashboard        # http://127.0.0.1:8765/
```

Notes on these steps:

- `make doctor` may report `[FAIL] SEGGER JLinkSWOViewerCL` on Linux, where
  SEGGER names the binary `JLinkSWOViewerCLExe`. That only affects `nsx view`;
  the dashboard finds either name.
- `make build` needs about 350 MB of disk in total (modules plus build tree).
- No board? `make replay` shows the dashboard on a recorded log, and
  `make test` runs the host tests. Neither needs the toolchain or J-Link.
- `make demo` is build + flash + dashboard in one command.

On the page: pick a word, press **Start attempt** (or Space), say it within
4 s. The verdict card turns green when the recognizer fires on that word.
The first ~2 s after reset are muted while the microphone settles
("Microphone settling" on the page).

## Expected SWO output

```
kws_i2s runtime=helia-rt arena_used=16820
perf_mode=LOW cpu_hz=96000000
i2s backend=hal fs=16000 bclk=1024000 slots=2x32 bits=24 shift=8 hpf=60 startup_hops=100
gpio clk=47 ws=48 sdin=4
model sha256=ae08012b5a5d…
always_on sleep=1 infer_lines=1 rtt_pcm=0 stride_ms=100
inf pred=<label> score=<s>                         10 per second
event label=<label> score=<s>                      when the recognizer fires
Frame <N>  peak=<n> dc=<n> overruns=0 dma_faults=0 slot=<0|1> slot_changes=<n> muted=<0|1> awake=<%>
stage frontend_cycles=<n> mfcc_cycles=<n> quantize_cycles=<n> invoke_cycles=<n>
live pred=<label> score=<s> fired=<0|1> invoke_cycles=<n> invoke_us=<n>
```

| Field | Meaning |
|---|---|
| `peak=` / `dc=` | last 20 ms hop handed to the model (int16, after the high-pass) |
| `slot=` | which word of the stereo pair carries the microphone |
| `muted=` | 1 during the start-up discard |
| `awake=` | share of the last second the core spent outside the sleep call |
| `stage` | DWT cycles: frontend per hop; log+DCT, quantize, Invoke per inference |

## How the always-on loop works

```
I2S0 + ping-pong DMA ─ interrupt every 20 ms (640 words = 320 frames)
  → slot pick → 60 Hz high-pass → 24→16 bit            audio/i2s_pcm16.h
  → PCM ring (last 1.11 s) + window/FFT/mel per hop    kws_core
  → feature ring (57 frames of mel energies)           kws_core
  → every 5 hops: clip peak → log → DCT → int8 → Invoke → recognizer
  → SWO lines; the core sleeps until the next interrupt
```

The KWS pipeline is `kws_core/`, a verbatim copy (see `PROVENANCE.md`); this
project adds the I2S source (`src/i2s_audio_source.c`), the signal
conditioning (`audio/i2s_pcm16.h`), the sleeping main loop (`src/main.cc`) and
the dashboard (`host/`).

## Build options

Set in the CMake cache after the first build, e.g.
`cmake -DKWS_I2S_RTT_PCM=ON build/apollo510_evb`, then `make build`.

| Option | Default | Effect |
|---|---|---|
| `KWS_I2S_SLEEP` | ON | sleep the core between hops |
| `KWS_I2S_INFER_LINES` | ON | one `inf` line per inference |
| `KWS_I2S_RTT_PCM` | OFF | dump the model's input hops on RTT channel 1 |
| `KWS_I2S_PCM_SHIFT` | 8 | right shift from 24-bit to int16 |
| `KWS_I2S_HPF_HZ` | 60 | high-pass cutoff, 0 disables |
| `KWS_I2S_STARTUP_HOPS` | 100 | hops muted after start (2 s) |

## Hear what the model hears

```bash
cmake -DKWS_I2S_RTT_PCM=ON build/apollo510_evb
make build flash
make capture          # capture/hop_pcm.wav, 16 kHz mono int16
```

Stop the dashboard first: the J-Link is exclusive.

## Tests

```bash
make test             # kws_core (88 GoogleTest cases), I2S conditioning, dashboard parser
```

No board, toolchain or `nsx` needed — only a host C++ compiler, CMake and
Python. The first `make test` downloads GoogleTest.

## Measured on the EVB (2026-10-09, this firmware, 96 MHz)

| Quantity | Value | Source |
|---|---|---|
| Hop rate | 50 per second, `overruns=0` | `Frame` lines |
| Frontend per hop | 117 817 cycles (1.23 ms) | `stage` line, DWT |
| log + DCT per inference | 684 336 cycles (7.13 ms) | `stage` line, DWT |
| Quantize per inference | 10 808 cycles (0.11 ms) | `stage` line, DWT |
| Invoke per inference | ≈ 990 200 cycles (10.31 ms) | `stage` line, DWT |
| Core awake | 25.7–25.8 % | `awake=`, DWT |
| Room background at the model input | about −68 dBFS RMS | RTT hop capture |
| Image | 296 048 B text, 2 724 B data, 510 304 B bss | `arm-none-eabi-size` |

These are cycle counts of this binary. They are not a power or energy
measurement: "asleep 74 % of the time" does not mean "74 % less current".

## What this demo does not claim

- A live-microphone `pred` is not an accuracy figure; the attempt tally on the
  dashboard is a bench check and is never written to `results/`.
- Cycle counts here are not comparable with heliaPROFILER or host numbers.
- The live result is not compared with the flash-clip or UART examples: the
  audio is different.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `ERROR: I2S source failed step=N hal_status=M` | HAL step N failed (list in `src/i2s_audio.h`) |
| "No live signal" on the page, `peak=0` | SD not on GPIO 4, no 1.8 V, or L/R floating |
| "Slot moved" warning | the sample changed word position; the firmware follows it |
| Dashboard says "J-Link error" | another SWO viewer or RTT tool holds the probe |
| `nsx: command not found` | activate the virtual environment from step 1 |
| `arm-none-eabi-gcc` not found during configure | toolchain `bin/` is not on `PATH` (step 2) |
| `make flash`: "Cannot connect to the probe" | EVB not connected over USB, or another J-Link program is running |

## Notes for maintainers

- `main.cc` uses the full `am_bsp_low_power_init()` (`skip_bsp_init = false`).
  With the minimal init that `kws_pdm` uses, `am_hal_i2s_enable()` fails with
  `AM_HAL_STATUS_INVALID_OPERATION` (a clock-manager request is refused). The
  exact cause is not established: giving the clock manager the board crystal
  information and an HFRC configuration by hand did not fix it.
- `modules/`, `boards/`, `cmake/nsx/` and `build/` are not in git; `nsx`
  recreates them from `nsx.yml` + `nsx.lock`. Run `make lock` only after
  editing `nsx.yml`.
- A build in a different directory or with a different CMake gives a
  different binary hash: the heliaRT "release-with-logs" variant embeds source
  paths.
- Where the copied parts came from: `PROVENANCE.md`.
- Design plan and status (Russian): `docs/IMPLEMENTATION_PLAN.md`.
