#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Record the hop PCM that kws_i2s feeds to the model (RTT channel 1) to a WAV.

Channel 1 carries the mono int16 hops at 16 kHz after slot pick, high-pass
and the 24->16 bit shift, i.e. exactly what kws_core receives. Labels stay
on SWO. Firmware must be built with ``-DKWS_I2S_RTT_PCM=ON``.

    python3 host/rtt_pcm_dump.py --duration 8 --out capture/voice.wav

Stop the dashboard / any SWO viewer first: the J-Link OB is exclusive.
This is a check of what the microphone delivers, not a KWS measurement.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

DEVICE = "AP510NFA-CBR"
SAMPLE_RATE_HZ = 16000
# DTCM, where the linker places the RTT control block.
RTT_SEARCH = "0x20000000 0x80000"
JLINK_DIR = "/opt/SEGGER/JLink"
# The firmware's RTT buffer (8 hops of 320 int16): audio parked there before
# the logger attached is older than the capture and is dropped.
STALE_BYTES = 8 * 320 * 2
ATTACH_S = 1.5


def record_raw(duration_s: float) -> bytes:
    """Run JLinkRTTLogger on channel 1 for *duration_s* and return its bytes."""
    logger = shutil.which("JLinkRTTLoggerExe") or shutil.which("JLinkRTTLoggerExe", path=JLINK_DIR)
    if logger is None:
        raise SystemExit("JLinkRTTLoggerExe not found (PATH or /opt/SEGGER/JLink)")
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "rtt.bin"
        # The logger quits on any key, including EOF: give it a pipe we never write.
        proc = subprocess.Popen(
            [logger, "-Device", DEVICE, "-If", "SWD", "-Speed", "4000", "-RTTChannel", "1",
             "-RTTSearchRanges", RTT_SEARCH, str(raw)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True)
        try:
            time.sleep(duration_s + ATTACH_S)
        finally:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait()
        return raw.read_bytes()[STALE_BYTES:] if raw.exists() else b""


def decode_pcm16(raw: bytes) -> list[int]:
    """Return the int16 samples in *raw* (little-endian), dropping a trailing odd byte."""
    n = len(raw) // 2
    return list(struct.unpack(f"<{n}h", raw[: 2 * n]))


def dbfs(samples: list[int]) -> float:
    """RMS level of int16 samples in dB re full scale, mean removed."""
    if not samples:
        return float("-inf")
    mean = sum(samples) / len(samples)
    rms = math.sqrt(sum((s - mean) ** 2 for s in samples) / len(samples))
    return 20 * math.log10(rms / 32768) if rms > 0 else float("-inf")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--out", type=Path, default=Path("capture/hop_pcm.wav"))
    args = parser.parse_args()

    samples = decode_pcm16(record_raw(args.duration))
    if not samples:
        print("no RTT data: is kws_i2s built with -DKWS_I2S_RTT_PCM=ON and the J-Link free?",
              file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.out), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE_HZ)
        wav.writeframes(struct.pack(f"<{len(samples)}h", *samples))

    seconds = len(samples) / SAMPLE_RATE_HZ
    print(f"{args.out}: {len(samples)} samples = {seconds:.2f} s at {SAMPLE_RATE_HZ} Hz")
    print(f"min={min(samples)} max={max(samples)} mean={sum(samples) / len(samples):.1f} "
          f"distinct={len(set(samples))} rms={dbfs(samples):.1f} dBFS")
    if len(set(samples)) < 3:
        print("WARNING: the stream is constant - no signal reaches the model", file=sys.stderr)
    print("RMS per second (dBFS):",
          " ".join(f"{dbfs(samples[i:i + SAMPLE_RATE_HZ]):.1f}"
                   for i in range(0, len(samples) - SAMPLE_RATE_HZ + 1, SAMPLE_RATE_HZ)))
    print("peak per second:",
          " ".join(str(max(abs(s) for s in samples[i:i + SAMPLE_RATE_HZ]))
                   for i in range(0, len(samples) - SAMPLE_RATE_HZ + 1, SAMPLE_RATE_HZ)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
