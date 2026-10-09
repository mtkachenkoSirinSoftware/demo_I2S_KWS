// SPDX-License-Identifier: Apache-2.0
#include "model/quantizer.h"

#include <gtest/gtest.h>

#include <cmath>

#include "model/model_runner.h"

using kws::DequantizeInt8;
using kws::ModelRunner;
using kws::QuantizeInt8;

// The parameters are read from the .tflite, never assumed.
TEST(Quantizer, UsesModelParametersNotDefaults) {
  EXPECT_NE(KWS_INPUT_SCALE, 1.0f);
  EXPECT_NE(KWS_INPUT_ZERO_POINT, 0);
  EXPECT_NE(KWS_OUTPUT_SCALE, 1.0f);
  EXPECT_NE(KWS_OUTPUT_ZERO_POINT, 0);
}

TEST(Quantizer, ZeroMapsToZeroPoint) {
  EXPECT_EQ(QuantizeInt8(0.0f, KWS_INPUT_SCALE, KWS_INPUT_ZERO_POINT), KWS_INPUT_ZERO_POINT);
}

TEST(Quantizer, RoundTripWithinHalfAStep) {
  for (float v = -30.0f; v <= 20.0f; v += 0.25f) {
    const int8_t q = QuantizeInt8(v, KWS_INPUT_SCALE, KWS_INPUT_ZERO_POINT);
    const float back = DequantizeInt8(q, KWS_INPUT_SCALE, KWS_INPUT_ZERO_POINT);
    // Only meaningful inside the representable range.
    const float lo = DequantizeInt8(-128, KWS_INPUT_SCALE, KWS_INPUT_ZERO_POINT);
    const float hi = DequantizeInt8(127, KWS_INPUT_SCALE, KWS_INPUT_ZERO_POINT);
    if (v < lo || v > hi) continue;
    EXPECT_LE(std::fabs(back - v), KWS_INPUT_SCALE * 0.5f + 1e-4f) << "v=" << v;
  }
}

TEST(Quantizer, SaturatesInsteadOfWrapping) {
  EXPECT_EQ(QuantizeInt8(1e9f, KWS_INPUT_SCALE, KWS_INPUT_ZERO_POINT), 127);
  EXPECT_EQ(QuantizeInt8(-1e9f, KWS_INPUT_SCALE, KWS_INPUT_ZERO_POINT), -128);
  // A wrapping bug would turn a large positive into a negative int8.
  EXPECT_GT(QuantizeInt8(500.0f, KWS_INPUT_SCALE, KWS_INPUT_ZERO_POINT), 0);
}

TEST(Quantizer, MatchesExplicitAffineFormula) {
  const float v = -3.75f;
  const int expected = static_cast<int>(std::round(v / KWS_INPUT_SCALE)) + KWS_INPUT_ZERO_POINT;
  EXPECT_EQ(QuantizeInt8(v, KWS_INPUT_SCALE, KWS_INPUT_ZERO_POINT),
            static_cast<int8_t>(expected));
}

TEST(Quantizer, SoftmaxIsNormalisedAndOrderPreserving) {
  int8_t logits[KWS_NUM_CLASSES];
  for (int i = 0; i < KWS_NUM_CLASSES; ++i) logits[i] = static_cast<int8_t>(-50 + i * 7);
  float out[KWS_NUM_CLASSES];
  kws::SoftmaxInt8(logits, KWS_NUM_CLASSES, KWS_OUTPUT_SCALE, KWS_OUTPUT_ZERO_POINT, out);

  float sum = 0.0f;
  for (int i = 0; i < KWS_NUM_CLASSES; ++i) {
    EXPECT_GE(out[i], 0.0f);
    sum += out[i];
  }
  EXPECT_NEAR(sum, 1.0f, 1e-5f);
  for (int i = 1; i < KWS_NUM_CLASSES; ++i) EXPECT_GT(out[i], out[i - 1]);
}

TEST(Quantizer, SoftmaxHandlesSaturatedLogitsWithoutOverflow) {
  int8_t logits[KWS_NUM_CLASSES];
  for (int i = 0; i < KWS_NUM_CLASSES; ++i) logits[i] = -128;
  logits[3] = 127;
  float out[KWS_NUM_CLASSES];
  kws::SoftmaxInt8(logits, KWS_NUM_CLASSES, KWS_OUTPUT_SCALE, KWS_OUTPUT_ZERO_POINT, out);
  for (int i = 0; i < KWS_NUM_CLASSES; ++i) EXPECT_TRUE(std::isfinite(out[i]));
  EXPECT_NEAR(out[3], 1.0f, 1e-3f);
}

TEST(ModelRunner, LabelOrderIsTfdsNotMlperf) {
  // tfds speech_commands order. MLPerf Tiny starts with silence/unknown and
  // puts go at index 11.
  EXPECT_STREQ(kws::kLabels[0], "down");
  EXPECT_STREQ(kws::kLabels[1], "go");
  EXPECT_STREQ(kws::kLabels[9], "yes");
  EXPECT_STREQ(kws::kLabels[10], "_silence_");
  EXPECT_STREQ(kws::kLabels[11], "_unknown_");
}

TEST(ModelRunner, SetInputQuantizesWholeWindow) {
  ModelRunner m;
  ASSERT_TRUE(m.Init());
  float mfcc[KWS_NUM_FRAMES * KWS_NUM_MFCC];
  for (size_t i = 0; i < KWS_NUM_FRAMES * KWS_NUM_MFCC; ++i) {
    mfcc[i] = -10.0f + 0.01f * static_cast<float>(i);
  }
  ASSERT_TRUE(m.SetInput(mfcc));
  for (size_t i = 0; i < KWS_NUM_FRAMES * KWS_NUM_MFCC; ++i) {
    EXPECT_EQ(m.input_tensor()[i],
              QuantizeInt8(mfcc[i], KWS_INPUT_SCALE, KWS_INPUT_ZERO_POINT)) << "at " << i;
  }
}

TEST(ModelRunner, InputTensorSizeMatchesModelContract) {
  // [1, 49, 10, 1] int8 from the .tflite.
  EXPECT_EQ(KWS_NUM_FRAMES * KWS_NUM_MFCC, 490);
}

TEST(ModelRunner, GetOutputPicksArgmax) {
  ModelRunner m;
  ASSERT_TRUE(m.Init());
  int8_t* out = m.mutable_output_tensor();
  for (int i = 0; i < KWS_NUM_CLASSES; ++i) out[i] = -40;
  out[6] = 100;
  const kws::Prediction p = m.GetOutput();
  EXPECT_EQ(p.label, 6);
  EXPECT_GT(p.score, 0.9f);
}

// Guards the one thing that makes every other number in this binary readable:
// which runtime produced it. The host test build links the stub backend, so no
// accuracy or timing figure from `kws_tests` describes an inference. If this
// assertion ever fails, the test binary has been linked against a real runtime
// and its results mean something different — update the reports, do not update
// the assertion.
TEST(ModelRunner, HostTestsRunWithoutAnInferenceRuntime) {
  EXPECT_EQ(kws::ModelRunner::runtime(), kws::Runtime::kHostStub);
  EXPECT_STREQ(kws::ModelRunner::runtime_name(), "host-stub");
  EXPECT_EQ(kws::ModelRunner::arena_used_bytes(), 0u);
}

// The generated contract is the source of truth for scales/ZPs. Its absence
// is allowed (a fresh checkout has not embedded a model yet) but must be
// visible, not silent. The SHA pin is the *shipping* model; a local Class B
// swap dirties this assertion on purpose.
TEST(ModelRunner, EmbeddedModelContractIsPresentAndConsistent) {
#ifdef KWS_HAVE_MODEL_CONTRACT
  EXPECT_EQ(KWS_MODEL_INPUT_ELEMENTS, KWS_NUM_FRAMES * KWS_NUM_MFCC);
  EXPECT_EQ(KWS_MODEL_OUTPUT_ELEMENTS, KWS_NUM_CLASSES);
  EXPECT_EQ(KWS_INPUT_ZERO_POINT, KWS_MODEL_INPUT_ZERO_POINT);
  EXPECT_EQ(KWS_OUTPUT_ZERO_POINT, KWS_MODEL_OUTPUT_ZERO_POINT);
  EXPECT_EQ(KWS_MODEL_NUM_OPS, 6);  // PAD, CONV_2D, DW_CONV_2D, AVGPOOL, RESHAPE, FC
  EXPECT_STREQ(KWS_MODEL_SHA256,
               "ae08012b5a5dd1fd59673bba3d4b1d1c43d7cd1f410537824d7eb6c2edb40fb7");
#else
  GTEST_SKIP() << "no model embedded yet; run tools/embed_model.py";
#endif
}
