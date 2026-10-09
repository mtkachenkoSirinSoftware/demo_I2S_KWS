// SPDX-License-Identifier: Apache-2.0
// heliaRT backend.
//
// heliaRT is a drop-in LiteRT for Micro / TFLM runtime, so this file is
// ordinary TFLM code: GetModel -> MicroMutableOpResolver -> MicroInterpreter ->
// AllocateTensors -> Invoke. What makes it *helia* is the link, not the source:
// nsx::helia_rt built with NSX_HELIA_RT_BACKEND=helia routes CONV_2D,
// DEPTHWISE_CONV_2D, FULLY_CONNECTED, AVERAGE_POOL_2D and RESHAPE onto
// heliaCORE (ns-cmsis-nn) MVE kernels. Switching to NSX_HELIA_RT_BACKEND=
// cmsis_nn or reference changes performance and nothing else about this file.
//
// Verified against helia-rt 1.19.0 (`cfab1523`):
//   tensorflow/lite/micro/micro_interpreter.h
//   tensorflow/lite/micro/micro_mutable_op_resolver.h
#include "model/model_runner.h"

#if defined(KWS_RUNTIME_HELIA_RT)

#include <new>

#include "model/generated/kws_model_contract.h"
#include "model/kws_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#if defined(KWS_CLIP_PMU)
#include "nsx_core.h"
#include "nsx_pmu_profiler.h"
#endif

// Arena placement.
//
// Where the arena lives dominates inference cost on Apollo510 far more than
// most kernel choices: TCM is single-cycle, SSRAM is not. NSX exposes the
// placement as a storage attribute from nsx_mem.h (nsx-core), which is how
// heliaPROFILER's own generated firmware places its arena. It is opt-in here
// because the host build has no NSX headers.
#if defined(KWS_USE_NSX_MEM)
#include "nsx_mem.h"
#if defined(KWS_ARENA_IN_SRAM)
#define KWS_ARENA_ATTR NSX_MEM_SRAM_BSS
#else
#define KWS_ARENA_ATTR NSX_MEM_FAST_BSS  // TCM
#endif
#else
#define KWS_ARENA_ATTR  // plain .bss; wherever the link script puts it
#endif

namespace kws {
namespace {

// KWS_MODEL_NUM_OPS is generated from the model's own operator table, so the
// resolver is never oversized (each slot costs flash) and never undersized
// (which surfaces as a misleading AllocateTensors failure).
using KwsOpResolver = tflite::MicroMutableOpResolver<KWS_MODEL_NUM_OPS>;

// All backend state is static: no heap, no steady-state allocation.
// alignas(16) matches heliaPROFILER's generated firmware and keeps the arena
// clear of Cortex-M55 cache-line / MVE load boundaries.
KWS_ARENA_ATTR alignas(16) uint8_t g_arena[KWS_TENSOR_ARENA_BYTES];

// Aligned storage rather than raw pointers, so the interpreter's lifetime is
// the program's and nothing is constructed on the inference path.
alignas(alignof(tflite::MicroInterpreter)) uint8_t g_interpreter_storage[sizeof(tflite::MicroInterpreter)];
tflite::MicroInterpreter* g_interpreter = nullptr;
KwsOpResolver g_resolver;
uint32_t g_arena_used = 0;
#if defined(KWS_CLIP_PMU)
NsxPmuProfiler g_clip_pmu;
#endif

}  // namespace

bool ModelRunner::Init() {
  // Protocol RESET must not rebuild the interpreter. Placement-new +
  // AllocateTensors on the same arena a second time hangs on Apollo510.
  if (initialised_ && g_interpreter != nullptr) return true;

  input_ = nullptr;
  output_ = nullptr;
  last_error_ = 0;
  initialised_ = false;

  tflite::InitializeTarget();

  const tflite::Model* model = tflite::GetModel(g_kws_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    last_error_ = 0xE5C0u;
    return false;
  }

  static bool ops_registered = false;
  if (!ops_registered) {
    KWS_MODEL_REGISTER_OPS(g_resolver);
    ops_registered = true;
  }

#if defined(KWS_CLIP_PMU)
  g_clip_pmu.Init(NSX_PMU_PRESET_ML_DEFAULT);
  g_interpreter = new (g_interpreter_storage) tflite::MicroInterpreter(
      model, g_resolver, g_arena, sizeof(g_arena), nullptr, &g_clip_pmu);
#else
  g_interpreter = new (g_interpreter_storage)
      tflite::MicroInterpreter(model, g_resolver, g_arena, sizeof(g_arena));
#endif

  const TfLiteStatus status = g_interpreter->AllocateTensors();
  if (status != kTfLiteOk) {
    // Ops are known-registered (the resolver list is generated from this exact
    // model), so the remaining causes are an undersized arena or a kernel
    // Prepare() rejecting a shape. Both are visible in arena_used_bytes().
    last_error_ = static_cast<uint32_t>(status);
    return false;
  }
  g_arena_used = static_cast<uint32_t>(g_interpreter->arena_used_bytes());

  TfLiteTensor* in = g_interpreter->input(0);
  TfLiteTensor* out = g_interpreter->output(0);
  if (in == nullptr || out == nullptr) {
    last_error_ = 0xE10u;
    return false;
  }
  // The generated contract already asserts dtype and element count against
  // kws_config.h at compile time; this catches the case where the *linked*
  // model data is not the one the contract was generated from.
  if (in->type != kTfLiteInt8 || out->type != kTfLiteInt8 ||
      in->bytes != KWS_NUM_FRAMES * KWS_NUM_MFCC ||
      out->bytes != KWS_NUM_CLASSES) {
    last_error_ = 0xE11u;
    return false;
  }

  input_ = in->data.int8;
  output_ = out->data.int8;
  initialised_ = true;
  return true;
}

bool ModelRunner::Invoke() {
  if (!initialised_ || g_interpreter == nullptr) return false;
#if defined(KWS_CLIP_PMU)
  g_clip_pmu.ClearEvents();
#endif
  const TfLiteStatus status = g_interpreter->Invoke();
  if (status != kTfLiteOk) {
    last_error_ = static_cast<uint32_t>(status);
    return false;
  }
  return true;
}

Runtime ModelRunner::runtime() { return Runtime::kHeliaRt; }
const char* ModelRunner::runtime_name() { return "helia-rt"; }
uint32_t ModelRunner::arena_used_bytes() { return g_arena_used; }

void ModelRunner::PrintClipPmuCsv() {
#if defined(KWS_CLIP_PMU)
  nsx_printf("--- Per-Layer PMU ---\n");
  nsx_printf(
      "  model-only, this binary; not always-on milliwatts; not hpx profile; "
      "last Invoke on synthetic PCM\n");
  g_clip_pmu.PrintCsv();
  if (g_interpreter != nullptr) {
    g_clip_pmu.BeginWholeInvokeMeasure();
    (void)g_interpreter->Invoke();
    g_clip_pmu.EndWholeInvokeMeasure();
    nsx_printf("dwt_cycles=%lu pmu_cycles=%lu inst_retired=%lu\n",
               (unsigned long)g_clip_pmu.dwt_cycles(),
               (unsigned long)g_clip_pmu.pmu_cycles(),
               (unsigned long)g_clip_pmu.inst_retired());
    nsx_printf(
        "  pmu_profiling events on a second Invoke of the same tensor; "
        "invoke_cycles= is first Invoke CYCCNT (with per-layer PMU). "
        "A mismatch is a fact, not a bug.\n");
  }
#endif
}

}  // namespace kws

#endif  // KWS_RUNTIME_HELIA_RT
