# SPDX-License-Identifier: Apache-2.0
# Label order vs neuralspotx/examples/kws_infer

Firmware here prints **tfds** names. Do not remap classes in C to match
`kws_infer`. Embedding a class-B array into `kws_infer` leaves their
`kLabels[]` until that table is edited.

Both graphs are 12-way INT8. The **index of a name is not shared**.

| Index | This tree (tfds `speech_commands`) | `kws_infer` (MLPerf Tiny) |
|---|---|---|
| 0 | `down` | `silence` |
| 1 | **`go`** | `unknown` |
| 2 | `left` | `yes` |
| 3 | `no` | `no` |
| 4 | `off` | `up` |
| 5 | `on` | `down` |
| 6 | `right` | `left` |
| 7 | `stop` | `right` |
| 8 | `up` | `on` |
| 9 | `yes` | `off` |
| 10 | `_silence_` | `stop` |
| 11 | `_unknown_` | **`go`** |

Clip/UART identity: interpreter `pred=go` is **index 1** on SHA `ae08012b…`.
In `kws_infer` the string `"go"` is **index 11**. Dummy-input `kws_infer`
does not run this PCM.

Source: `kws_core/model/model_runner.cc` (`kLabels`) vs
`neuralspotx/examples/kws_infer/src/main.cc`.
