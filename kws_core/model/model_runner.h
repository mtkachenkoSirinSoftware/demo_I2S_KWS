// SPDX-License-Identifier: Apache-2.0
// Runtime-agnostic inference facade.
//
// The concrete backend is chosen at compile time — no vtable, no heap, nothing
// resolved at runtime. Exactly one of these must be defined:
//
//   KWS_RUNTIME_HOST_STUB   host tests. No inference happens; the output tensor
//                           is writable so the quantisation, post-processing and
//                           application paths are still exercised end to end.
//                           This is the default when nothing is defined.
//   KWS_RUNTIME_HELIA_RT    heliaRT — Ambiq's LiteRT for Micro / TFLM runtime,
//                           kernels from heliaCORE (ns-cmsis-nn) via
//                           NSX_HELIA_RT_BACKEND=helia. Consumes the .tflite
//                           embedded by tools/embed_model.py.
//   KWS_RUNTIME_EXECUTORCH  nsx-executorch — NSX adapter for ExecuTorch
//                           Cortex-M. Consumes a .pte, not a .tflite.
//
// ns-cmsis-nn is deliberately absent from that list: it is a *kernel library*,
// not a runtime (its own README: "Not a model runtime"). It is reached through
// heliaRT's helia backend or through ExecuTorch's CMSIS-NN provider, never
// linked directly by this application.
//
// Backend state is a singleton held inside the backend translation unit: the
// arena, interpreter and model pointer are file-static there, so this class
// stays a thin handle and the header pulls in no runtime headers. The
// application creates exactly one ModelRunner (app/kws_app.cc), which is what
// makes that legitimate.
#ifndef MODEL_RUNNER_H_
#define MODEL_RUNNER_H_

#include <stddef.h>
#include <stdint.h>

#include "config/kws_config.h"

#if !defined(KWS_RUNTIME_HOST_STUB) && !defined(KWS_RUNTIME_HELIA_RT) && \
    !defined(KWS_RUNTIME_EXECUTORCH)
#define KWS_RUNTIME_HOST_STUB 1
#endif
#if (defined(KWS_RUNTIME_HOST_STUB) + defined(KWS_RUNTIME_HELIA_RT) + \
     defined(KWS_RUNTIME_EXECUTORCH)) > 1
#error "define exactly one of KWS_RUNTIME_HOST_STUB / _HELIA_RT / _EXECUTORCH"
#endif

// Tensor arena / working-set budget, in bytes.
//
// 32 KiB is the ceiling heliaPROFILER uses for unpruned MLPerf-Tiny KWS, so it
// is a ceiling for this r=0.60 graph rather than a fitted size.
//
// What is actually known: linking this model against helia_rt::reference and
// calling AllocateTensors() on the host reports arena_used_bytes() == 12352.
// Arena planning is a property of the graph, not of the CPU, so that number is
// analytically determined rather than measured — but it is a LOWER BOUND, not
// the answer, for two reasons:
//
//   * It was produced by the *reference* kernels, which request no scratch at
//     all (tensorflow/lite/micro/kernels/conv.cc: zero
//     RequestScratchBufferInArena calls). The heliaCORE kernels this firmware
//     actually links request scratch in conv, depthwise_conv and
//     fully_connected (4, 4 and 3 call sites respectively), so the HELIA arena
//     is strictly larger. How much larger is not knowable off-target.
//   * Alignment and head/tail padding differ per toolchain.
//
// So: keep the ceiling until an Apollo510 build reports its own figure, then
// shrink to that plus headroom. The backend logs it at boot
// (ModelRunner::arena_used_bytes() on SWO). `hpx profile` reports a different
// figure (HPX_ALLOCATED_ARENA) from other firmware.
#ifndef KWS_TENSOR_ARENA_BYTES
#define KWS_TENSOR_ARENA_BYTES (32 * 1024)
#endif

namespace kws {

struct Prediction {
  int label = -1;
  float score = 0.0f;
  float scores[KWS_NUM_CLASSES] = {};
};

extern const char* const kLabels[KWS_NUM_CLASSES];

// Which backend this binary was built with (SWO `runtime=`).
enum class Runtime : uint8_t { kHostStub = 0, kHeliaRt, kExecuTorch };

class ModelRunner {
 public:
  // Brings up the backend: parses the embedded model, plans the arena and
  // binds the I/O tensor pointers. Returns false and leaves last_error() set
  // if any of that fails — a failed Init is never papered over, because the
  // alternative is a demo that reports confident nonsense.
  bool Init();

  // Quantizes `mfcc` (KWS_NUM_FRAMES * KWS_NUM_MFCC floats) into the input
  // tensor. Kept separate from Invoke() so the quantization step is testable
  // on its own.
  bool SetInput(const float* mfcc);

  bool Invoke();
  Prediction GetOutput() const;

  // These point *into the backend's own tensors* (the heliaRT arena, the
  // ExecuTorch I/O buffers, or the host stub's static arrays) — there is no
  // staging copy on the inference path.
  const int8_t* input_tensor() const { return input_; }
  int8_t* mutable_output_tensor() { return output_; }

  static Runtime runtime();
  static const char* runtime_name();

  // Bytes of arena the backend actually consumed, once Init() has run.
  // 0 when the backend cannot report it. This is a host-observable *memory*
  // fact, not a latency one, so it is safe to publish as measured.
  static uint32_t arena_used_bytes();

  // kws_clip only: per-layer ML_DEFAULT CSV plus pmu_profiling
  // (CPU_CYCLES + INST_RETIRED) on a second Invoke of the same tensor.
  // No-op without KWS_CLIP_PMU. Not hpx profile, not always-on milliwatts.
  static void PrintClipPmuCsv();

  // Backend-specific status from the last failed call; 0 when clean.
  // heliaRT: TfLiteStatus. ExecuTorch: (stage << 16) | executorch_error.
  uint32_t last_error() const { return last_error_; }

 private:
  int8_t* input_ = nullptr;
  int8_t* output_ = nullptr;
  uint32_t last_error_ = 0;
  bool initialised_ = false;
};

}  // namespace kws
#endif  // MODEL_RUNNER_H_
