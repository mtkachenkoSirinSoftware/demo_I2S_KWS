#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Embed an INT8 .tflite as C arrays for the Apollo510 examples.

Emits, from one model file:

  kws_model_data.cc        the flatbuffer as an aligned C array
  kws_model_contract.h     shape / dtype / quantisation / op set / resolver
                           registrations, read out of the flatbuffer
  kws_model_manifest.json  SHA256, sizes, ops (local; not shipped in the tarball)

``kws_config.h`` consumes the generated scales and zero points. Re-quantise,
re-run this script, rebuild. Class C (wrong shape / dtype / class count) fails
``--check`` instead of silently mis-scaling.

Enums and ``MicroMutableOpResolver`` method names come from a helia-rt
checkout (``nsx lock`` → ``modules/helia-rt``, or ``--helia-rt``). TensorFlow
is not required.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import struct
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Minimal flatbuffer reader (read-only, enough for the TFLite schema)
# --------------------------------------------------------------------------


class Buf:
    """Flatbuffer accessor. All offsets are absolute byte positions."""

    def __init__(self, data: bytes) -> None:
        self.d = data

    def u8(self, p: int) -> int:
        return self.d[p]

    def i8(self, p: int) -> int:
        return struct.unpack_from("<b", self.d, p)[0]

    def u16(self, p: int) -> int:
        return struct.unpack_from("<H", self.d, p)[0]

    def i32(self, p: int) -> int:
        return struct.unpack_from("<i", self.d, p)[0]

    def u32(self, p: int) -> int:
        return struct.unpack_from("<I", self.d, p)[0]

    def i64(self, p: int) -> int:
        return struct.unpack_from("<q", self.d, p)[0]

    def f32(self, p: int) -> float:
        return struct.unpack_from("<f", self.d, p)[0]

    def indirect(self, p: int) -> int:
        """Follow a uoffset stored at `p` (relative to `p`)."""
        return p + self.u32(p)


class Table:
    """A flatbuffer table; `field(i)` returns the absolute position or None."""

    def __init__(self, buf: Buf, pos: int) -> None:
        self.b = buf
        self.pos = pos
        self.vtable = pos - buf.i32(pos)
        self.vtable_size = buf.u16(self.vtable)

    def field(self, idx: int) -> int | None:
        off = 4 + 2 * idx
        if off >= self.vtable_size:
            return None
        voff = self.b.u16(self.vtable + off)
        return None if voff == 0 else self.pos + voff

    # -- scalars ----------------------------------------------------------
    def u8_f(self, idx: int, default: int = 0) -> int:
        p = self.field(idx)
        return default if p is None else self.b.u8(p)

    def i8_f(self, idx: int, default: int = 0) -> int:
        p = self.field(idx)
        return default if p is None else self.b.i8(p)

    def i32_f(self, idx: int, default: int = 0) -> int:
        p = self.field(idx)
        return default if p is None else self.b.i32(p)

    def u32_f(self, idx: int, default: int = 0) -> int:
        p = self.field(idx)
        return default if p is None else self.b.u32(p)

    # -- reference types --------------------------------------------------
    def string_f(self, idx: int) -> str | None:
        p = self.field(idx)
        if p is None:
            return None
        s = self.b.indirect(p)
        n = self.b.u32(s)
        return self.b.d[s + 4 : s + 4 + n].decode("utf-8", "replace")

    def table_f(self, idx: int) -> "Table | None":
        p = self.field(idx)
        return None if p is None else Table(self.b, self.b.indirect(p))

    def vector(self, idx: int) -> tuple[int, int]:
        """(elements_start, length); (0, 0) when the field is absent."""
        p = self.field(idx)
        if p is None:
            return (0, 0)
        v = self.b.indirect(p)
        return (v + 4, self.b.u32(v))

    def i32_vec(self, idx: int) -> list[int]:
        start, n = self.vector(idx)
        return [self.b.i32(start + 4 * i) for i in range(n)]

    def f32_vec(self, idx: int) -> list[float]:
        start, n = self.vector(idx)
        return [self.b.f32(start + 4 * i) for i in range(n)]

    def i64_vec(self, idx: int) -> list[int]:
        start, n = self.vector(idx)
        return [self.b.i64(start + 8 * i) for i in range(n)]

    def table_vec(self, idx: int) -> list["Table"]:
        start, n = self.vector(idx)
        return [Table(self.b, self.b.indirect(start + 4 * i)) for i in range(n)]

    def ubyte_vec_len(self, idx: int) -> int:
        _, n = self.vector(idx)
        return n


# TFLite schema field indices. Source of truth:
#   helia-rt tensorflow/lite/schema/schema_generated.h
# (field order in a flatbuffer table is part of the format and never reordered
#  for an existing schema, only appended to.)
F_MODEL_VERSION, F_MODEL_OPCODES, F_MODEL_SUBGRAPHS = 0, 1, 2
F_MODEL_DESCRIPTION, F_MODEL_BUFFERS = 3, 4
F_OPCODE_DEPRECATED, F_OPCODE_CUSTOM, F_OPCODE_VERSION, F_OPCODE_BUILTIN = 0, 1, 2, 3
F_SG_TENSORS, F_SG_INPUTS, F_SG_OUTPUTS, F_SG_OPERATORS, F_SG_NAME = 0, 1, 2, 3, 4
F_T_SHAPE, F_T_TYPE, F_T_BUFFER, F_T_NAME, F_T_QUANT = 0, 1, 2, 3, 4
F_Q_MIN, F_Q_MAX, F_Q_SCALE, F_Q_ZERO_POINT = 0, 1, 2, 3
F_OP_OPCODE_INDEX, F_OP_INPUTS, F_OP_OUTPUTS = 0, 1, 2
F_BUF_DATA = 0


# --------------------------------------------------------------------------
# heliaRT-derived lookup tables
# --------------------------------------------------------------------------


# Official TFLite TensorType (schema field order is append-only). Used only by
# inspect_io / --check when helia-rt is absent. Full embed still reads the
# enums and Add* names from the helia-rt checkout so they cannot go stale.
_TENSOR_TYPES_FALLBACK = {
    0: "FLOAT32",
    1: "FLOAT16",
    2: "INT32",
    3: "UINT8",
    4: "INT64",
    5: "STRING",
    6: "BOOL",
    7: "INT16",
    8: "COMPLEX64",
    9: "INT8",
    10: "FLOAT64",
    11: "COMPLEX128",
    12: "UINT64",
    13: "RESOURCE",
    14: "VARIANT",
    15: "UINT32",
    16: "UINT16",
    17: "INT4",
}

# Firmware I/O this pipeline can run without a rewrite.
FIRMWARE_INPUT_SHAPE = [1, 49, 10, 1]
FIRMWARE_OUTPUT_SHAPE = [1, 12]
FIRMWARE_IO_TYPE = "INT8"

# Shipping-model quant identity. Class A matches this; Class B matches geometry
# but not these scales / zero points / SHA.
SHIPPING_SHA256 = "ae08012b5a5dd1fd59673bba3d4b1d1c43d7cd1f410537824d7eb6c2edb40fb7"
SHIPPING_INPUT_SCALE = 0.5899888277053833
SHIPPING_INPUT_ZERO_POINT = 81
SHIPPING_OUTPUT_SCALE = 0.20843878388404846
SHIPPING_OUTPUT_ZERO_POINT = 42


def find_helia_rt(explicit: Path | None, start: Path) -> Path | None:
    """Return the helia-rt checkout or None if it is not on disk."""
    if explicit is not None:
        return explicit if explicit.is_dir() else None
    schema = Path("tensorflow") / "lite" / "schema" / "schema_generated.h"
    for parent in [start, *start.parents]:
        candidates = (
            parent / "third_party" / "helia-rt",
            parent / "modules" / "helia-rt",
            parent / "kws_clip" / "modules" / "helia-rt",
            parent / "kws_uart" / "modules" / "helia-rt",
            parent / "kws_pdm" / "modules" / "helia-rt",
            parent / "ambiq_kws_examples" / "kws_clip" / "modules" / "helia-rt",
        )
        for cand in candidates:
            if (cand / schema).is_file():
                return cand
    return None


def _find_helia_rt(explicit: Path | None, start: Path) -> Path:
    found = find_helia_rt(explicit, start)
    if found is not None:
        return found
    sys.exit(
        "error: could not locate helia-rt. TFLite enums and MicroMutableOpResolver\n"
        "       names are read from that checkout. Pass --helia-rt <path>, run\n"
        "       `nsx lock` under kws_clip/ (modules/helia-rt), or clone\n"
        "       https://github.com/AmbiqAI/helia-rt @ cfab1523."
    )


def _parse_enum(header: Path, name: str) -> dict[int, str]:
    src = header.read_text()
    m = re.search(rf"enum {name}[^{{]*\{{(.*?)\n\}};", src, re.S)
    if m is None:
        sys.exit(f"error: enum {name} not found in {header}")
    out: dict[int, str] = {}
    for member, value in re.findall(rf"{name}_([A-Z0-9_]+)\s*=\s*(-?\d+)", m.group(1)):
        # MIN/MAX sentinels alias real members; first definition wins.
        out.setdefault(int(value), member)
    return out


def _parse_resolver_methods(header: Path) -> set[str]:
    src = header.read_text()
    return set(re.findall(r"\bTfLiteStatus\s+Add([A-Za-z0-9_]+)\s*\(", src))


def _resolver_method_for(op_name: str, available: set[str]) -> str:
    """CONV_2D -> AddConv2D, verified against the real resolver header."""
    candidates = []
    tokens = op_name.split("_")
    candidates.append("".join(t if t[:1].isdigit() else t.capitalize() for t in tokens))
    candidates.append("".join(t.capitalize() for t in tokens))
    candidates.append("".join(t.title() for t in tokens))
    for c in candidates:
        if c in available:
            return f"Add{c}"
    # Case-insensitive last resort, so e.g. LSTM-style acronyms still resolve.
    lowered = {a.lower(): a for a in available}
    for c in candidates:
        if c.lower() in lowered:
            return f"Add{lowered[c.lower()]}"
    sys.exit(
        f"error: no MicroMutableOpResolver::Add* method matches op '{op_name}'.\n"
        f"       Tried {candidates}. This op cannot be registered with heliaRT's\n"
        f"       MicroMutableOpResolver; it needs a custom registration."
    )


# --------------------------------------------------------------------------
# Model inspection
# --------------------------------------------------------------------------


def describe_tensor(t: Table, tensor_types: dict[int, str]) -> dict:
    shape = t.i32_vec(F_T_SHAPE)
    q = t.table_f(F_T_QUANT)
    scales = q.f32_vec(F_Q_SCALE) if q else []
    zeros = q.i64_vec(F_Q_ZERO_POINT) if q else []
    n = 1
    for d in shape:
        n *= d
    return {
        "name": t.string_f(F_T_NAME),
        "shape": shape,
        "elements": n,
        "type": tensor_types.get(t.u8_f(F_T_TYPE), f"UNKNOWN({t.u8_f(F_T_TYPE)})"),
        "scale": scales[0] if len(scales) == 1 else scales,
        "zero_point": zeros[0] if len(zeros) == 1 else zeros,
    }


def _open_tflite(path: Path) -> tuple[bytes, Table]:
    data = path.read_bytes()
    if data[4:8] not in (b"TFL3", b"TFL2"):
        sys.exit(
            f"error: {path} does not carry a TFLite file identifier "
            f"(got {data[4:8]!r}); refusing to guess the format."
        )
    b = Buf(data)
    return data, Table(b, b.indirect(0))


def inspect_io(path: Path, tensor_types: dict[int, str] | None = None) -> dict:
    """I/O contract only. Does not need helia-rt (uses the stable TensorType enum)."""
    types = tensor_types if tensor_types is not None else _TENSOR_TYPES_FALLBACK
    data, model = _open_tflite(path)
    subgraphs = model.table_vec(F_MODEL_SUBGRAPHS)
    if len(subgraphs) != 1:
        sys.exit(
            f"error: {path} has {len(subgraphs)} subgraphs; the firmware runner "
            "assumes a single-subgraph model."
        )
    sg = subgraphs[0]
    tensors = sg.table_vec(F_SG_TENSORS)
    inputs = sg.i32_vec(F_SG_INPUTS)
    outputs = sg.i32_vec(F_SG_OUTPUTS)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "schema_version": model.u32_f(F_MODEL_VERSION),
        "n_subgraphs": 1,
        "inputs": [describe_tensor(tensors[i], types) for i in inputs],
        "outputs": [describe_tensor(tensors[i], types) for i in outputs],
    }


def _quant_scalar(v: object) -> float | int | None:
    if isinstance(v, list):
        return None
    return v  # type: ignore[return-value]


def load_sidecar(model: Path, explicit: Path | None = None) -> dict | None:
    """Sibling ``*.helia.json`` if present (optional unstructured-sidecar check)."""
    if explicit is not None:
        if not explicit.is_file():
            sys.exit(f"error: sidecar {explicit} not found")
        return json.loads(explicit.read_text())
    for cand in (model.with_name(model.stem + ".helia.json"), model.with_suffix(".helia.json")):
        if cand.is_file():
            return json.loads(cand.read_text())
    return None


def classify(info: dict, sidecar: dict | None = None, allow_unstructured: bool = False) -> dict:
    """Class A / B / C / reject against the firmware geometry.

    A — drop-in: int8 [1,49,10,1] → [1,12], same scales as the shipping model.
    B — contract-only: same geometry, different scales / SHA (re-embed + rebuild).
    C — needs firmware: shape, dtype, class count, or per-axis quant differ.
    U — unstructured / factorized sidecar: not a heliaCORE width prune
        (``can_embed`` only with ``allow_unstructured``).
    """
    reasons: list[str] = []
    if len(info.get("inputs", [])) != 1 or len(info.get("outputs", [])) != 1:
        return {
            "class": "C",
            "label": "needs-firmware",
            "can_embed": False,
            "reasons": ["firmware assumes exactly one input and one output tensor"],
        }
    inp, out = info["inputs"][0], info["outputs"][0]
    in_scale = _quant_scalar(inp.get("scale"))
    in_zp = _quant_scalar(inp.get("zero_point"))
    out_scale = _quant_scalar(out.get("scale"))
    out_zp = _quant_scalar(out.get("zero_point"))

    if in_scale is None or in_zp is None:
        reasons.append("input is per-axis quantised; runner assumes per-tensor I/O")
    if out_scale is None or out_zp is None:
        reasons.append("output is per-axis quantised; runner assumes per-tensor I/O")
    if inp.get("type") != FIRMWARE_IO_TYPE:
        reasons.append(f"input type {inp.get('type')} != {FIRMWARE_IO_TYPE}")
    if out.get("type") != FIRMWARE_IO_TYPE:
        reasons.append(f"output type {out.get('type')} != {FIRMWARE_IO_TYPE}")
    if inp.get("shape") != FIRMWARE_INPUT_SHAPE:
        reasons.append(f"input shape {inp.get('shape')} != {FIRMWARE_INPUT_SHAPE}")
    if out.get("shape") != FIRMWARE_OUTPUT_SHAPE:
        reasons.append(f"output shape {out.get('shape')} != {FIRMWARE_OUTPUT_SHAPE}")
    if reasons:
        return {
            "class": "C",
            "label": "needs-firmware",
            "can_embed": False,
            "reasons": reasons,
        }

    side = sidecar or {}
    axis = str(side.get("compression_axis") or "")
    if axis in {"weight_sparsity", "tucker_factorized"} and not allow_unstructured:
        return {
            "class": "U",
            "label": "not-helia-structural",
            "can_embed": False,
            "reasons": [
                side.get("reason")
                or f"sidecar compression_axis={axis}; heliaRT runs dense MVE kernels"
            ],
        }

    same_quant = (
        in_scale == SHIPPING_INPUT_SCALE
        and in_zp == SHIPPING_INPUT_ZERO_POINT
        and out_scale == SHIPPING_OUTPUT_SCALE
        and out_zp == SHIPPING_OUTPUT_ZERO_POINT
    )
    same_sha = info.get("sha256") == SHIPPING_SHA256
    extras: list[str] = []
    if side.get("helium_lane_aligned") is False:
        extras.append(
            "sidecar widths are not multiples of 8 — heliaRT will run them, "
            "Helium lanes may pad; not a measured cycle win"
        )
    if same_quant:
        extra = extras + (
            [] if same_sha else ["weights/SHA differ from the shipping model; scales match"]
        )
        return {
            "class": "A",
            "label": "drop-in",
            "can_embed": True,
            "reasons": extra or ["matches shipping geometry and quant identity"],
        }
    return {
        "class": "B",
        "label": "contract-only",
        "can_embed": True,
        "reasons": extras
        + [
            "geometry matches; quant parameters differ — embed regenerates "
            "kws_model_contract.h and kws_config.h follows it"
        ],
    }


def inspect(path: Path, helia_rt: Path) -> dict:
    schema_h = helia_rt / "tensorflow" / "lite" / "schema" / "schema_generated.h"
    resolver_h = helia_rt / "tensorflow" / "lite" / "micro" / "micro_mutable_op_resolver.h"
    if not resolver_h.is_file():
        sys.exit(f"error: {resolver_h} not found")
    builtin_ops = _parse_enum(schema_h, "BuiltinOperator")
    tensor_types = _parse_enum(schema_h, "TensorType")
    resolver_methods = _parse_resolver_methods(resolver_h)

    data, model = _open_tflite(path)

    opcodes = []
    for oc in model.table_vec(F_MODEL_OPCODES):
        # GetBuiltinCode(): max(builtin_code, deprecated_builtin_code)
        # — tensorflow/compiler/mlir/lite/schema/schema_utils.cc
        code = max(oc.i32_f(F_OPCODE_BUILTIN, 0), oc.i8_f(F_OPCODE_DEPRECATED, 0))
        opcodes.append(
            {
                "code": code,
                "name": builtin_ops.get(code, f"UNKNOWN({code})"),
                "custom_code": oc.string_f(F_OPCODE_CUSTOM),
                "version": oc.i32_f(F_OPCODE_VERSION, 1),
            }
        )

    subgraphs = model.table_vec(F_MODEL_SUBGRAPHS)
    if len(subgraphs) != 1:
        sys.exit(
            f"error: {path} has {len(subgraphs)} subgraphs; the firmware runner "
            "assumes a single-subgraph model."
        )
    sg = subgraphs[0]
    tensors = sg.table_vec(F_SG_TENSORS)
    operators = sg.table_vec(F_SG_OPERATORS)
    buffers = model.table_vec(F_MODEL_BUFFERS)

    op_sequence = [opcodes[o.u32_f(F_OP_OPCODE_INDEX)]["name"] for o in operators]
    # Registration order does not matter to TFLM; sorting keeps the generated
    # header stable across re-exports that reorder the graph.
    unique_ops = sorted(set(op_sequence))

    weight_bytes = 0
    for t in tensors:
        buf_idx = t.u32_f(F_T_BUFFER)
        if buf_idx < len(buffers):
            weight_bytes += buffers[buf_idx].ubyte_vec_len(F_BUF_DATA)

    inputs = sg.i32_vec(F_SG_INPUTS)
    outputs = sg.i32_vec(F_SG_OUTPUTS)

    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "schema_version": model.u32_f(F_MODEL_VERSION),
        "description": model.string_f(F_MODEL_DESCRIPTION),
        "num_tensors": len(tensors),
        "num_operators": len(operators),
        "weight_bytes": weight_bytes,
        "inputs": [describe_tensor(tensors[i], tensor_types) for i in inputs],
        "outputs": [describe_tensor(tensors[i], tensor_types) for i in outputs],
        "op_sequence": op_sequence,
        "ops": unique_ops,
        "custom_ops": sorted(
            {o["custom_code"] for o in opcodes if o["name"] == "CUSTOM" and o["custom_code"]}
        ),
        "resolver_registrations": [
            f"{_resolver_method_for(op, resolver_methods)}();" for op in unique_ops
        ],
    }


# --------------------------------------------------------------------------
# Emitters
# --------------------------------------------------------------------------

_SPDX = "// SPDX-License-Identifier: Apache-2.0\n"


def _banner(model_path: Path, info: dict, tool: str) -> str:
    return (
        f"// GENERATED FILE — do not edit.\n"
        f"// Produced by {tool} from:\n"
        f"//   {model_path.name}\n"
        f"//   SHA256 {info['sha256']}\n"
        f"//   {info['size_bytes']} bytes, TFLite schema v{info['schema_version']}\n"
    )


def emit_model_data(info: dict, model_path: Path, data: bytes, symbol: str, tool: str) -> str:
    lines = [
        _SPDX,
        _banner(model_path, info, tool),
        "//\n"
        "// Alignment: TFLM requires the flatbuffer to be at least 4-byte aligned\n"
        "// (tflite::GetModel dereferences it in place); 16 keeps it clear of the\n"
        "// Cortex-M55 cache-line and MVE load boundaries too.\n",
        "\n",
        '#include "model/kws_model_data.h"\n',
        "\n",
        f"alignas(16) const unsigned char {symbol}[] = {{\n",
    ]
    for i in range(0, len(data), 16):
        chunk = data[i : i + 16]
        lines.append("    " + " ".join(f"0x{c:02x}," for c in chunk) + "\n")
    lines.append("};\n")
    lines.append(f"const unsigned int {symbol}_len = {len(data)};\n")
    return "".join(lines)


def emit_kws_infer_header(info: dict, model_path: Path, data: bytes, tool: str) -> str:
    """Drop-in replacement for neuralspotx/examples/kws_infer/src/kws_model_data.h.

    Symbols match their ``main.cc`` (``kws_model_data`` / ``kws_model_data_len``).
    Does not rewrite their ``kLabels[]`` — that table stays classic MLPerf until
    edited; see assets/LABELS.md.
    """
    lines = [
        "#ifndef KWS_MODEL_DATA_H\n",
        "#define KWS_MODEL_DATA_H\n",
        "#include <cstdint>\n",
        "\n",
        f"// Auto-generated by {tool} from {model_path.name} ({len(data)} bytes)\n",
        f"// SHA256 {info['sha256']}\n",
        "// Do not hand-edit this array. Replace the whole header.\n",
        "// kLabels[] in kws_infer/src/main.cc stays MLPerf (go = index 11)\n",
        "// until that table is edited. See ambiq_kws_examples/assets/LABELS.md.\n",
        "alignas(16) unsigned char kws_model_data[] = {\n",
    ]
    for i in range(0, len(data), 12):
        chunk = data[i : i + 12]
        lines.append("    " + " ".join(f"0x{c:02x}," for c in chunk) + "\n")
    lines.append("};\n")
    lines.append(f"const unsigned int kws_model_data_len = {len(data)};\n")
    lines.append("#endif\n")
    return "".join(lines)


def emit_contract(info: dict, model_path: Path, tool: str, guard: str) -> str:
    inp, out = info["inputs"][0], info["outputs"][0]

    def _scalar(v, what):
        if isinstance(v, list):
            sys.exit(
                f"error: {what} is per-axis quantised ({len(v)} values); the firmware "
                "runner assumes per-tensor quantisation on the model I/O."
            )
        return v

    in_scale = _scalar(inp["scale"], "input scale")
    in_zp = _scalar(inp["zero_point"], "input zero point")
    out_scale = _scalar(out["scale"], "output scale")
    out_zp = _scalar(out["zero_point"], "output zero point")

    ops = "\n".join(f"//   {o}" for o in info["op_sequence"])
    regs = "\n".join(f"    r.{reg}" for reg in info["resolver_registrations"])
    regs_continued = regs.replace("\n", " \\\n")
    body = f"""{_SPDX}{_banner(model_path, info, tool)}//
// Everything below was read out of the flatbuffer, never assumed. Include it
// from config/kws_config.h — that header *consumes* these scales/zero points
// so a Class B re-quantised model does not require hand-editing literals.
//
// Graph, in execution order:
{ops}
#ifndef {guard}
#define {guard}

#define KWS_MODEL_SHA256 "{info['sha256']}"
#define KWS_MODEL_BYTES  {info['size_bytes']}
#define KWS_MODEL_SCHEMA_VERSION {info['schema_version']}
#define KWS_MODEL_NUM_OPERATORS  {info['num_operators']}
#define KWS_MODEL_NUM_TENSORS    {info['num_tensors']}
// Sum of every constant buffer: the flash cost of the weights alone.
#define KWS_MODEL_WEIGHT_BYTES   {info['weight_bytes']}

// Distinct builtin ops -> the MicroMutableOpResolver<N> size the runner needs.
#define KWS_MODEL_NUM_OPS {len(info['ops'])}

// --- input tensor: {inp['name']} ---
#define KWS_MODEL_INPUT_ELEMENTS   {inp['elements']}
#define KWS_MODEL_INPUT_TYPE_{inp['type']} 1
#define KWS_MODEL_INPUT_SCALE      {in_scale!r}f
#define KWS_MODEL_INPUT_ZERO_POINT {in_zp}

// --- output tensor: {out['name']} ---
#define KWS_MODEL_OUTPUT_ELEMENTS   {out['elements']}
#define KWS_MODEL_OUTPUT_TYPE_{out['type']} 1
#define KWS_MODEL_OUTPUT_SCALE      {out_scale!r}f
#define KWS_MODEL_OUTPUT_ZERO_POINT {out_zp}

// Registration list for tflite::MicroMutableOpResolver<KWS_MODEL_NUM_OPS>.
// Method names were resolved against heliaRT's own
// tensorflow/lite/micro/micro_mutable_op_resolver.h — an op with no Add*
// method fails generation rather than reaching AllocateTensors() as a
// mis-read "arena too small".
#define KWS_MODEL_REGISTER_OPS(r) \\
    do {{ \\
{regs_continued} \\
    }} while (0)

#endif  // {guard}
"""
    return body


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", type=Path, help="path to the .tflite")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "model" / "generated",
        help="directory for the generated files (default: model/generated)",
    )
    ap.add_argument("--symbol", default="g_kws_model_data", help="C array symbol name")
    ap.add_argument("--helia-rt", type=Path, default=None, help="path to helia-rt (nsx lock → modules/helia-rt)")
    ap.add_argument(
        "--inspect-only",
        action="store_true",
        help="print the model contract as JSON and write nothing",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="classify the model (A/B embeddable, C I/O, U unstructured sidecar) "
        "and write nothing. Exit 0 for A/B, 2 for C/U.",
    )
    ap.add_argument("--sidecar", type=Path, default=None, help="path to a .helia.json sidecar")
    ap.add_argument(
        "--allow-unstructured",
        action="store_true",
        help="embed a weight-sparsity sidecar (heliaCORE still runs dense convs)",
    )
    ap.add_argument(
        "--kws-infer-header",
        type=Path,
        default=None,
        help="write a drop-in neuralspotx/examples/kws_infer/src/kws_model_data.h "
        "(does not edit their kLabels[])",
    )
    args = ap.parse_args()

    if not args.model.is_file():
        sys.exit(f"error: {args.model} not found")

    sidecar = load_sidecar(args.model.resolve(), args.sidecar)

    if args.check:
        info = inspect_io(args.model.resolve())
        verdict = classify(info, sidecar=sidecar, allow_unstructured=args.allow_unstructured)
        payload = {**info, "compatibility": verdict, "helia_sidecar": sidecar}
        json.dump(payload, sys.stdout, indent=2)
        print()
        print(
            f"class {verdict['class']} ({verdict['label']}): "
            + "; ".join(verdict["reasons"]),
            file=sys.stderr,
        )
        return 0 if verdict["can_embed"] else 2

    helia_rt = _find_helia_rt(args.helia_rt, args.model.resolve().parent)
    info = inspect(args.model.resolve(), helia_rt)

    if args.inspect_only:
        info["compatibility"] = classify(
            info, sidecar=sidecar, allow_unstructured=args.allow_unstructured
        )
        info["helia_sidecar"] = sidecar
        json.dump(info, sys.stdout, indent=2)
        print()
        return 0

    verdict = classify(info, sidecar=sidecar, allow_unstructured=args.allow_unstructured)
    if not verdict["can_embed"]:
        print(
            f"error: class {verdict['class']} ({verdict['label']}): "
            + "; ".join(verdict["reasons"]),
            file=sys.stderr,
        )
        return 2

    tool = f"tools/{Path(__file__).name}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = args.model.read_bytes()

    (args.out_dir / "kws_model_data.cc").write_text(
        emit_model_data(info, args.model, data, args.symbol, tool)
    )
    (args.out_dir / "kws_model_contract.h").write_text(
        emit_contract(info, args.model, tool, "KWS_MODEL_CONTRACT_H_")
    )

    manifest = dict(info)
    manifest["generated_by"] = tool
    manifest["generated_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    manifest["helia_rt_path"] = str(helia_rt)
    manifest["helia_rt_describe"] = _git_describe(helia_rt)
    manifest["symbol"] = args.symbol
    manifest["compatibility"] = verdict
    manifest["helia_sidecar"] = sidecar
    (args.out_dir / "kws_model_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if args.kws_infer_header is not None:
        args.kws_infer_header.parent.mkdir(parents=True, exist_ok=True)
        args.kws_infer_header.write_text(emit_kws_infer_header(info, args.model, data, tool))
        print(f"wrote {args.kws_infer_header}  (kws_infer drop-in; kLabels[] unchanged)")

    verdict = manifest["compatibility"]
    print(f"wrote {args.out_dir}/kws_model_data.cc      ({len(data)} bytes of model)")
    print(f"wrote {args.out_dir}/kws_model_contract.h")
    print(f"wrote {args.out_dir}/kws_model_manifest.json")
    print(f"  ops        : {', '.join(info['ops'])}")
    print(f"  input      : {info['inputs'][0]['shape']} {info['inputs'][0]['type']}")
    print(f"  output     : {info['outputs'][0]['shape']} {info['outputs'][0]['type']}")
    print(f"  weights    : {info['weight_bytes']} bytes")
    print(f"  class      : {verdict['class']} ({verdict['label']})")
    return 0


def _git_describe(repo: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "describe", "--tags", "--always", "--dirty"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
