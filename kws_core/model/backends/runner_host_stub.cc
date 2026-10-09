// SPDX-License-Identifier: Apache-2.0
// Host backend: no inference runtime at all.
//
// This is not a fake inference. Invoke() computes nothing and says so; the
// output tensor is plain writable memory so tests can drive a known logit
// vector through GetOutput(), the recognizer and the application loop. Any
// accuracy or latency claim from a binary built this way is meaningless, and
// runtime_name() reports "host-stub" precisely so such a claim cannot be made
// by accident.
#include "model/model_runner.h"

#if defined(KWS_RUNTIME_HOST_STUB)

namespace kws {
namespace {
// Same storage the real backends hand out from their arenas, so the pointer
// semantics of input_tensor() / mutable_output_tensor() are identical.
int8_t g_input[KWS_NUM_FRAMES * KWS_NUM_MFCC];
int8_t g_output[KWS_NUM_CLASSES];
}  // namespace

bool ModelRunner::Init() {
  for (size_t i = 0; i < sizeof(g_input); ++i) g_input[i] = 0;
  for (size_t i = 0; i < sizeof(g_output); ++i) g_output[i] = 0;
  input_ = g_input;
  output_ = g_output;
  last_error_ = 0;
  initialised_ = true;
  return true;
}

bool ModelRunner::Invoke() { return initialised_; }

Runtime ModelRunner::runtime() { return Runtime::kHostStub; }
const char* ModelRunner::runtime_name() { return "host-stub"; }
uint32_t ModelRunner::arena_used_bytes() { return 0; }
void ModelRunner::PrintClipPmuCsv() {}

}  // namespace kws

#endif  // KWS_RUNTIME_HOST_STUB
