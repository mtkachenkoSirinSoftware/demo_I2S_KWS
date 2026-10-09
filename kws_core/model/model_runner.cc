// SPDX-License-Identifier: Apache-2.0
// Backend-independent half of ModelRunner. Init(), Invoke() and the static
// runtime accessors live in model/backends/<backend>.cc; everything here is
// identical whichever runtime is linked, which is the point of the seam.
#include "model/model_runner.h"

#include "model/quantizer.h"

namespace kws {

// tfds speech_commands label order. Not classic MLPerf — see assets/LABELS.md.
const char* const kLabels[KWS_NUM_CLASSES] = {
    "down", "go", "left", "no", "off", "on",
    "right", "stop", "up", "yes", "_silence_", "_unknown_"};

bool ModelRunner::SetInput(const float* mfcc) {
  if (!initialised_ || mfcc == nullptr || input_ == nullptr) return false;
  for (size_t i = 0; i < KWS_NUM_FRAMES * KWS_NUM_MFCC; ++i) {
    input_[i] = QuantizeInt8(mfcc[i], KWS_INPUT_SCALE, KWS_INPUT_ZERO_POINT);
  }
  return true;
}

Prediction ModelRunner::GetOutput() const {
  Prediction p;
  if (output_ == nullptr) return p;
  SoftmaxInt8(output_, KWS_NUM_CLASSES, KWS_OUTPUT_SCALE, KWS_OUTPUT_ZERO_POINT, p.scores);
  int best = 0;
  for (int i = 1; i < KWS_NUM_CLASSES; ++i) {
    if (p.scores[i] > p.scores[best]) best = i;
  }
  p.label = best;
  p.score = p.scores[best];
  return p;
}

}  // namespace kws
