// SPDX-License-Identifier: Apache-2.0
#include "model/quantizer.h"

#include <stddef.h>

namespace kws {

void SoftmaxInt8(const int8_t* logits, size_t n, float scale, int32_t zero_point, float* out) {
  if (logits == nullptr || out == nullptr || n == 0) return;
  float max_v = DequantizeInt8(logits[0], scale, zero_point);
  for (size_t i = 1; i < n; ++i) {
    const float v = DequantizeInt8(logits[i], scale, zero_point);
    if (v > max_v) max_v = v;
  }
  float sum = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    out[i] = expf(DequantizeInt8(logits[i], scale, zero_point) - max_v);
    sum += out[i];
  }
  const float inv = (sum > 0.0f) ? 1.0f / sum : 0.0f;
  for (size_t i = 0; i < n; ++i) out[i] *= inv;
}

}  // namespace kws
