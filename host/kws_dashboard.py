#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Live browser dashboard for kws_i2s: did the I2S microphone trigger a keyword?

Reads the firmware's **SWO** lines, parses ``event label=``, ``inf pred=``,
``live pred=``, ``Frame N  peak=`` and ``stage ...``, and pushes them to
``kws_dashboard.html`` over Server-Sent Events.

    python3 host/kws_dashboard.py --reset      # attach to the EVB, reset so the banner prints
    python3 host/kws_dashboard.py --replay host/fixtures/kws_i2s_swo.txt   # no board

Stop any other SWO viewer / ``rtt_pcm_dump.py`` first: the J-Link OB is
exclusive.

Cycle counts shown on the page (``invoke_cycles``, ``stage``, ``awake``) are
the firmware's own DWT prints, relayed unchanged. Hit / miss tallies on the
page are an informal bench check of a live microphone, not an accuracy
measurement on a dataset split. Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, TextIO

DEFAULT_DEVICE = "AP510NFA-CBR"
# Same values the generated ``kws_i2s_view`` target passes (perf_mode=LOW).
DEFAULT_CPU_HZ = 96000000
DEFAULT_SWO_HZ = 1000000
SWO_VIEWER_NAMES = ("JLinkSWOViewerCL", "JLinkSWOViewer_CL", "JLinkSWOViewerCLExe")
JLINK_DIR = Path("/opt/SEGGER/JLink")
HTML_PATH = Path(__file__).with_name("kws_dashboard.html")
HISTORY_LINES = 3000
KEEPALIVE_S = 10.0

_KV = re.compile(r"(\w+)=(\S+)")
_FRAME = re.compile(r"Frame\s+(\d+)\s+(peak=.*)")
_BANNER_KEYS = ("kws_i2s runtime=", "kws_pdm runtime=", "perf_mode=", "i2s backend=",
                "pdm backend=", "gpio clk=", "model sha256=", "always_on sleep=")
# SWO occasionally drops a newline, gluing two records into one line; a record
# is cut wherever one of these starts.
_RECORD_START = re.compile(
    r"(?=inf pred=|live pred=|event label=|Frame\s+\d+\s+peak=|stage frontend_cycles=|ERROR:)")


def _num(text: str) -> int | float | str:
    """Return *text* as int, then float, else unchanged."""
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text


def _fields(text: str) -> dict[str, int | float | str]:
    return {k: _num(v) for k, v in _KV.findall(text)}


def split_records(line: str) -> list[str]:
    """Split one SWO line into records, undoing dropped newlines."""
    return [part for part in _RECORD_START.split(line.strip()) if part.strip()]


def parse_record(text: str) -> dict[str, Any] | None:
    """Classify one record printed by ``demo_I2S/src/main.cc``.

    Parameters
    ----------
    text : str
        One record: a whole SWO line, or one piece from ``split_records``.

    Returns
    -------
    dict or None
        ``{"type": "event" | "inf" | "live" | "frame" | "stage" | "banner" |
        "error", ...}`` with the record's ``key=value`` fields, or ``None``.
    """
    text = text.strip()
    if text.startswith("event label="):
        return {"type": "event", **_fields(text)}
    if text.startswith("inf pred="):
        return {"type": "inf", **_fields(text)}
    if text.startswith("live pred="):
        return {"type": "live", **_fields(text)}
    if text.startswith("stage frontend_cycles="):
        return {"type": "stage", **_fields(text)}
    match = _FRAME.match(text)
    if match:
        return {"type": "frame", "frame": int(match.group(1)), **_fields(match.group(2))}
    # SWO can drop the very first byte after reset ("ws_i2s runtime=").
    if text.startswith(_BANNER_KEYS) or "runtime=" in text[:16]:
        return {"type": "banner", **_fields(text)}
    if text.startswith("ERROR:"):
        return {"type": "error", "message": text}
    return None


def parse_line(line: str) -> list[dict[str, Any]]:
    """Return every record found in one SWO line, in order."""
    out = []
    for part in split_records(line):
        msg = parse_record(part)
        if msg is not None:
            out.append(msg)
    return out


class Hub:
    """Fan parsed SWO messages out to every connected browser."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._clients: list[queue.Queue[str]] = []
        self._history: list[str] = []
        self._banner: dict[str, Any] = {}
        self._status: dict[str, Any] = {"type": "status", "state": "starting", "detail": ""}

    def _broadcast(self, payload: str, keep: bool) -> None:
        with self._lock:
            if keep:
                self._history.append(payload)
                del self._history[:-HISTORY_LINES]
            for client in self._clients:
                client.put(payload)

    def publish_line(self, line: str) -> None:
        """Parse *line*, stamp it with host time, and send it to the browsers."""
        now = time.time()
        for msg in parse_line(line):
            msg["t"] = now
            if msg["type"] == "banner":
                with self._lock:
                    self._banner.update({k: v for k, v in msg.items() if k not in ("type", "t")})
            self._broadcast(json.dumps(msg), keep=True)
        self._broadcast(json.dumps({"type": "raw", "t": now, "line": line.rstrip()}), keep=True)

    def set_status(self, state: str, detail: str = "") -> None:
        msg = {"type": "status", "state": state, "detail": detail, "t": time.time()}
        with self._lock:
            self._status = msg
        self._broadcast(json.dumps(msg), keep=False)

    def subscribe(self) -> tuple[queue.Queue[str], list[str]]:
        """Register a browser; return its queue plus the backlog to send first."""
        client: queue.Queue[str] = queue.Queue()
        with self._lock:
            backlog = [json.dumps({"type": "banner", **self._banner}), *self._history,
                       json.dumps(self._status)]
            self._clients.append(client)
        return client, backlog

    def unsubscribe(self, client: queue.Queue[str]) -> None:
        with self._lock:
            if client in self._clients:
                self._clients.remove(client)


def find_swo_viewer() -> str | None:
    """Locate SEGGER's command-line SWO viewer on PATH or under /opt/SEGGER."""
    for name in SWO_VIEWER_NAMES:
        found = shutil.which(name) or shutil.which(name, path=str(JLINK_DIR))
        if found:
            return found
    return None


def _reset_target(device: str) -> None:
    jlink = shutil.which("JLinkExe") or shutil.which("JLinkExe", path=str(JLINK_DIR))
    if jlink is None:
        print("JLinkExe not found; skipping --reset", file=sys.stderr)
        return
    subprocess.run(
        [jlink, "-nogui", "1", "-device", device, "-if", "SWD", "-speed", "4000"],
        input="Reset\nGo\nExit\n", text=True, stdout=subprocess.DEVNULL, check=False)


def run_swo(hub: Hub, args: argparse.Namespace, capture: TextIO | None) -> None:
    """Spawn the SWO viewer and feed its lines to *hub* until it exits."""
    viewer = find_swo_viewer()
    if viewer is None:
        hub.set_status("error", "JLinkSWOViewerCL not found (PATH or /opt/SEGGER/JLink)")
        return
    cmd = [viewer, "-device", args.device, "-cpufreq", str(args.cpufreq),
           "-swofreq", str(args.swofreq), "-itmport", "0"]
    stdbuf = shutil.which("stdbuf")
    if stdbuf:
        cmd = [stdbuf, "-oL", "-eL", *cmd]
    # The viewer quits on "any key", including EOF: hand it a pipe we never write.
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, start_new_session=True)
    args.viewer_proc = proc
    hub.set_status("attached", " ".join(cmd))
    if args.reset:
        time.sleep(0.4)
        _reset_target(args.device)
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.decode("utf-8", errors="replace")
        if capture is not None:
            capture.write(line)
            capture.flush()
        hub.publish_line(line)
    hub.set_status("error", f"SWO viewer exited (code {proc.wait()}); is the J-Link busy?")


def run_replay(hub: Hub, path: Path, speed: float) -> None:
    """Replay a captured SWO log at roughly real time.

    The firmware prints one ``inf`` line per 100 ms; a log without them is
    paced by its once-per-second ``Frame`` lines instead.
    """
    hub.set_status("replay", str(path))
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    has_inf = any("inf pred=" in line for line in lines)
    for line in lines:
        if has_inf:
            delay = 0.1 * line.count("inf pred=")
        else:
            delay = 1.0 if "Frame" in line else 0.0
        time.sleep(delay / speed)
        hub.publish_line(line)
    hub.set_status("replay-done", str(path))


def make_handler(hub: Hub) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *log_args: Any) -> None:  # noqa: A002
            return

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/events":
                self._events()
            elif self.path in ("/", "/index.html"):
                body = HTML_PATH.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)

        def _events(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            client, backlog = hub.subscribe()
            try:
                for payload in backlog:
                    self.wfile.write(f"data: {payload}\n\n".encode())
                self.wfile.flush()
                while True:
                    try:
                        payload = client.get(timeout=KEEPALIVE_S)
                        self.wfile.write(f"data: {payload}\n\n".encode())
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                hub.unsubscribe(client)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--cpufreq", type=int, default=DEFAULT_CPU_HZ)
    parser.add_argument("--swofreq", type=int, default=DEFAULT_SWO_HZ)
    parser.add_argument("--reset", action="store_true",
                        help="reset the target after attaching so the banner is printed")
    parser.add_argument("--capture", type=Path, help="also append raw SWO lines to FILE")
    parser.add_argument("--replay", type=Path, help="replay a captured SWO log (no board)")
    parser.add_argument("--speed", type=float, default=1.0, help="replay speed multiplier")
    args = parser.parse_args()
    args.viewer_proc = None

    hub = Hub()
    capture = args.capture.open("a", encoding="utf-8") if args.capture else None
    if args.replay:
        worker = threading.Thread(target=run_replay, args=(hub, args.replay, args.speed),
                                  daemon=True)
    else:
        worker = threading.Thread(target=run_swo, args=(hub, args, capture), daemon=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(hub))
    server.daemon_threads = True
    worker.start()
    # SIGTERM, and SIGINT when started as a background job (where the shell
    # leaves it ignored), must still reach the cleanup below.
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, signal.default_int_handler)
    print(f"kws_i2s dashboard: http://127.0.0.1:{args.port}/  (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        # Take the viewer's process group down, or it keeps holding the J-Link.
        if args.viewer_proc is not None and args.viewer_proc.poll() is None:
            os.killpg(args.viewer_proc.pid, signal.SIGTERM)
        if capture is not None:
            capture.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
