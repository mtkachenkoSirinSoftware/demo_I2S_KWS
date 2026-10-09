# kws_core

Portable pipeline: PCM hop → MFCC → int8 → ModelRunner → recognizer.
No UART HAL, no PDM, no NSX bootstrap. The three apps compile these
sources via `cmake/kws_core.cmake`.

```bash
cmake -S kws_core -B kws_core/build
cmake --build kws_core/build -j
ctest --test-dir kws_core/build --output-on-failure
```

Host cycle counters are **zero** (`profiling/kws_profile_host.cc`). That is
not an Apollo510 measurement.

| Path | Role |
|---|---|
| `dsp/` | streaming MFCC (tf.signal 30/20 ms) |
| `model/` | quantize + heliaRT runner |
| `postprocessing/` | smooth / threshold / debounce |
| `app/kws_app.*` | source-agnostic hop loop |
| `config/kws_config.h` | geometry + scales from the embedded contract |
