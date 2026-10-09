#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Dashboard SWO parser against the line shapes of demo_I2S/src/main.cc (no EVB)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "host"))

import kws_dashboard as dash  # noqa: E402

FIXTURE = ROOT / "host" / "fixtures" / "kws_i2s_swo.txt"


class ParseLineTest(unittest.TestCase):
    def one(self, line: str) -> dict:
        records = dash.parse_line(line)
        self.assertEqual(len(records), 1, line)
        return records[0]

    def test_banner_lines(self) -> None:
        self.assertEqual(self.one("kws_i2s runtime=helia-rt arena_used=16820"),
                         {"type": "banner", "runtime": "helia-rt", "arena_used": 16820})
        self.assertEqual(self.one("perf_mode=LOW cpu_hz=96000000")["cpu_hz"], 96000000)
        i2s = self.one("i2s backend=hal fs=16000 bclk=1024000 slots=2x32 bits=24 "
                       "shift=8 hpf=60 startup_hops=100")
        self.assertEqual((i2s["fs"], i2s["bclk"], i2s["slots"], i2s["hpf"]),
                         (16000, 1024000, "2x32", 60))
        self.assertEqual(self.one("gpio clk=47 ws=48 sdin=4"),
                         {"type": "banner", "clk": 47, "ws": 48, "sdin": 4})
        self.assertEqual(self.one("always_on sleep=1 infer_lines=1 rtt_pcm=0 stride_ms=100")["sleep"], 1)
        sha = "ae08012b5a5dd1fd59673bba3d4b1d1c43d7cd1f410537824d7eb6c2edb40fb7"
        self.assertEqual(self.one(f"model sha256={sha}")["sha256"], sha)

    def test_first_byte_dropped_after_reset(self) -> None:
        self.assertEqual(self.one("ws_i2s runtime=helia-rt arena_used=16820")["type"], "banner")

    def test_per_second_lines(self) -> None:
        frame = self.one("Frame 150  peak=20 dc=-3 overruns=0 dma_faults=0 slot=1 "
                         "slot_changes=2 muted=0 awake=25.7")
        self.assertEqual(frame, {"type": "frame", "frame": 150, "peak": 20, "dc": -3,
                                 "overruns": 0, "dma_faults": 0, "slot": 1, "slot_changes": 2,
                                 "muted": 0, "awake": 25.7})
        stage = self.one("stage frontend_cycles=117817 mfcc_cycles=684336 "
                         "quantize_cycles=10826 invoke_cycles=990220")
        self.assertEqual(stage["invoke_cycles"], 990220)
        live = self.one("live pred=_unknown_ score=0.247 fired=0 invoke_cycles=990336 invoke_us=10316")
        self.assertEqual((live["pred"], live["score"], live["invoke_us"]), ("_unknown_", 0.247, 10316))

    def test_inference_and_event(self) -> None:
        self.assertEqual(self.one("inf pred=yes score=0.912"),
                         {"type": "inf", "pred": "yes", "score": 0.912})
        self.assertEqual(self.one("event label=yes score=0.912"),
                         {"type": "event", "label": "yes", "score": 0.912})

    def test_glued_records(self) -> None:
        # SWO dropped the newline between two records (seen on the EVB).
        records = dash.parse_line("inf pred=on score=0.389inf pred=down score=0.352")
        self.assertEqual([(r["pred"], r["score"]) for r in records], [("on", 0.389), ("down", 0.352)])
        records = dash.parse_line("inf pred=yes score=0.900event label=yes score=0.900")
        self.assertEqual([r["type"] for r in records], ["inf", "event"])
        records = dash.parse_line("inf pred=go score=0.5Frame 200  peak=18 dc=-2 overruns=0 "
                                  "dma_faults=0 slot=0 slot_changes=0 muted=0 awake=25.8")
        self.assertEqual([r["type"] for r in records], ["inf", "frame"])
        self.assertEqual(records[0]["score"], 0.5)

    def test_error_and_noise(self) -> None:
        self.assertEqual(self.one("ERROR: i2s stalled, restart ok")["type"], "error")
        self.assertEqual(dash.parse_line("pred is the live I2S microphone, not a flash/UART clip"), [])
        self.assertEqual(dash.parse_line("Ambiq Secure BootLoader!"), [])
        self.assertEqual(dash.parse_line(""), [])

    def test_kws_pdm_lines_still_parse(self) -> None:
        self.assertEqual(self.one("kws_pdm runtime=helia-rt arena_used=16820")["type"], "banner")
        frame = self.one("Frame 50  peak=32680 dc=-183 overruns=0 dma_faults=0 derr=0")
        self.assertEqual((frame["peak"], frame["dc"]), (32680, -183))


class FixtureTest(unittest.TestCase):
    """A log recorded from the EVB: every structured line must parse."""

    def test_recorded_log(self) -> None:
        counts: dict[str, int] = {}
        for line in FIXTURE.read_text(encoding="utf-8").splitlines():
            for record in dash.parse_line(line):
                counts[record["type"]] = counts.get(record["type"], 0) + 1
        self.assertGreaterEqual(counts.get("banner", 0), 6)
        self.assertGreaterEqual(counts.get("frame", 0), 10)
        self.assertEqual(counts.get("frame"), counts.get("stage"))
        self.assertEqual(counts.get("frame"), counts.get("live"))
        # 10 inferences per second, give or take the first and last partial second.
        self.assertGreater(counts.get("inf", 0), 8 * counts["frame"])
        self.assertNotIn("error", counts)


if __name__ == "__main__":
    unittest.main()
