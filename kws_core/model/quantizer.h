// SPDX-License-Identifier: Apache-2.0
// INT8 affine conversion. Parameters come from the .tflite, never
// from an assumption of scale=1 / zero_point=0.
#ifndef QUANTIZER_H_
#define QUANTIZER_H_

#include <math.h>
#include <stdint.h>

#include "config/kws_config.h"

namespace kws {

// q = round(real / scale) + zero_point, saturated to int8.
inline int8_t QuantizeInt8(float real, float scale, int32_t zero_point) {
  float q = roundf(real / scale) + static_cast<float>(zero_point);
  if (q < -128.0f) q = -128.0f;
  if (q > 127.0f) q = 127.0f;
  return static_cast<int8_t>(q);
}

inline float DequantizeInt8(int8_t q, float scale, int32_t zero_point) {
  return (static_cast<float>(q) - static_cast<float>(zero_point)) * scale;
}

// Softmax over dequantized logits, in place over `n` values.
void SoftmaxInt8(const int8_t* logits, size_t n, float scale, int32_t zero_point, float* out);

}  // namespace kws
#endif  // QUANTIZER_H_
