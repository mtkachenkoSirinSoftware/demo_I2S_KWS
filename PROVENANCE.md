# Provenance

Where every non-original part of `demo_I2S` came from. Update this file in
the same change that touches a copied part.

## Copied from `ambiq_kws_examples`

Repository: <https://github.com/mtkachenkoSirinSoftware/Ambiqs-Zoo-of-Examples-Extension-.git>,
commit `129b8f7d140fa0a9b062db2768770296368854aa` (copied 2026-10-09).

| Here | Source | State |
|---|---|---|
| `kws_core/` | `kws_core/` (without `build/`) | byte-identical, **do not edit** |
| `assets/` | `assets/` | byte-identical |
| `tools/embed_model.py` | `tools/embed_model.py` | byte-identical |
| `model/generated/` | `kws_pdm/model/generated/` | byte-identical (output of `embed_model.py`) |
| `rtt/` | `kws_pdm/rtt/` | `SOURCE.md` wording adjusted |
| `src/kws_profile_target.c`, `src/apollo510_cache.h` | `kws_pdm/src/` | byte-identical |
| `LICENSE` | `LICENSE` | byte-identical |
| `nsx.yml` | `kws_pdm/nsx.yml` | project renamed, `nsx-audio` removed |
| `CMakeLists.txt`, `src/main.cc`, `src/kws_i2s_rtt.*` | `kws_pdm` equivalents | adapted |
| `host/kws_dashboard.*` | `kws_pdm/host/kws_dashboard.*` (uncommitted there) | adapted, English UI |

Check that the `kws_core` copy is still identical (with a checkout of that
repository next to this directory):

```bash
diff -rq --exclude=build ../ambiq_kws_examples/kws_core kws_core
```

## Model

`assets/depgraph_r060_kd_int8.tflite`, SHA256
`ae08012b5a5dd1fd59673bba3d4b1d1c43d7cd1f410537824d7eb6c2edb40fb7`,
18 256 bytes, input `[1,49,10,1]` int8, output `[1,12]` int8. The firmware
prints this SHA on SWO (`model sha256=`).

## nsx modules (`nsx.lock`)

| Module set | Revision |
|---|---|
| `nsx-ambiq-sdk` (HAL, BSP, core, power, …) | tag `v5.2.24`, `a9f4ec25a162f6f3700623feb691423bb5a51132` |
| `nsx-pmu-armv8m` | tag `v0.2.0`, `5725c065a0c3603132f1064ee2684d1fa8587c88` |
| `ns-cmsis-nn` | `b386770a4588e2b46a6ca6df97fa487c9f3dedd4` |
| `helia-rt` | `cfab1523a58bcfab69a106efea22c530b8100285` |

`modules/` is not in git: `nsx sync --app-dir .` (run by `make build`) downloads
these revisions. `neuralspotx` 0.7.17 was the tool version used.

## SEGGER RTT

`rtt/` — SEGGERMicro/RTT V8.58.0, commit
`4d8feab3150f86f37a9d323ddc88d6cdf5673072`; licence in `rtt/LICENSE.md`.

## I2S configuration

Pins, clock and format were established on an Apollo510 EVB with a
separate bring-up firmware, `i2s_mic_test` (AmbiqSuite 5.2.0 HAL, 2026-10-08) and re-verified with
this firmware on the nsx SDK revision above (2026-10-09).
