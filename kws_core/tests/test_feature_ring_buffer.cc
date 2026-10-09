// SPDX-License-Identifier: Apache-2.0
#include "dsp/feature_ring_buffer.h"

#include <gtest/gtest.h>

#include <vector>

using kws::FeatureRingBuffer;

namespace {
std::vector<float> MakeFrame(float tag) {
  std::vector<float> f(FeatureRingBuffer::kFrameWidth);
  for (size_t i = 0; i < f.size(); ++i) f[i] = tag + 0.001f * static_cast<float>(i);
  return f;
}
}  // namespace

TEST(FeatureRingBuffer, EmptyWindowNotReady) {
  FeatureRingBuffer r;
  r.Reset();
  EXPECT_FALSE(r.WindowReady());
  EXPECT_EQ(r.count(), 0u);
  EXPECT_EQ(r.FrameFromNewest(0), nullptr);
  EXPECT_EQ(r.WindowFrame(0), nullptr);
}

TEST(FeatureRingBuffer, BecomesReadyAtExactlyOneWindow) {
  FeatureRingBuffer r;
  r.Reset();
  for (int i = 0; i < KWS_NUM_FRAMES - 1; ++i) {
    auto f = MakeFrame(static_cast<float>(i));
    r.PushFrame(f.data());
    EXPECT_FALSE(r.WindowReady()) << "ready too early at " << i;
  }
  auto last = MakeFrame(static_cast<float>(KWS_NUM_FRAMES - 1));
  r.PushFrame(last.data());
  EXPECT_TRUE(r.WindowReady());
}

TEST(FeatureRingBuffer, FrameFromNewestIndexesBackwards) {
  FeatureRingBuffer r;
  r.Reset();
  for (int i = 0; i < 10; ++i) {
    auto f = MakeFrame(static_cast<float>(i));
    r.PushFrame(f.data());
  }
  EXPECT_FLOAT_EQ(r.FrameFromNewest(0)[0], 9.0f);
  EXPECT_FLOAT_EQ(r.FrameFromNewest(1)[0], 8.0f);
  EXPECT_FLOAT_EQ(r.FrameFromNewest(9)[0], 0.0f);
  EXPECT_EQ(r.FrameFromNewest(10), nullptr);
}

// The window must be exposed oldest-first without shifting history.
TEST(FeatureRingBuffer, WindowIsOldestFirst) {
  FeatureRingBuffer r;
  r.Reset();
  for (int i = 0; i < KWS_NUM_FRAMES; ++i) {
    auto f = MakeFrame(static_cast<float>(i));
    r.PushFrame(f.data());
  }
  ASSERT_TRUE(r.WindowReady());
  for (int i = 0; i < KWS_NUM_FRAMES; ++i) {
    ASSERT_NE(r.WindowFrame(i), nullptr);
    EXPECT_FLOAT_EQ(r.WindowFrame(i)[0], static_cast<float>(i)) << "at " << i;
  }
}

// The critical property: after wrapping many times the logical window is still
// the newest KWS_NUM_FRAMES frames, in order.
TEST(FeatureRingBuffer, WindowStaysCorrectAcrossManyWraps) {
  FeatureRingBuffer r;
  r.Reset();
  const int total = FeatureRingBuffer::kCapacityFrames * 7 + 3;
  for (int i = 0; i < total; ++i) {
    auto f = MakeFrame(static_cast<float>(i));
    r.PushFrame(f.data());
  }
  ASSERT_TRUE(r.WindowReady());
  const int oldest = total - KWS_NUM_FRAMES;
  for (int i = 0; i < KWS_NUM_FRAMES; ++i) {
    EXPECT_FLOAT_EQ(r.WindowFrame(i)[0], static_cast<float>(oldest + i)) << "at " << i;
  }
  EXPECT_EQ(r.total_frames(), static_cast<uint64_t>(total));
  EXPECT_EQ(r.count(), FeatureRingBuffer::kCapacityFrames);
}

TEST(FeatureRingBuffer, SlidingByOneFrameShiftsWindowByOne) {
  FeatureRingBuffer r;
  r.Reset();
  for (int i = 0; i < KWS_NUM_FRAMES; ++i) {
    auto f = MakeFrame(static_cast<float>(i));
    r.PushFrame(f.data());
  }
  const float first_before = r.WindowFrame(0)[0];
  auto extra = MakeFrame(static_cast<float>(KWS_NUM_FRAMES));
  r.PushFrame(extra.data());
  EXPECT_FLOAT_EQ(r.WindowFrame(0)[0], first_before + 1.0f);
  EXPECT_FLOAT_EQ(r.WindowFrame(KWS_NUM_FRAMES - 1)[0], static_cast<float>(KWS_NUM_FRAMES));
}

TEST(FeatureRingBuffer, WholeFrameContentsRoundTrip) {
  FeatureRingBuffer r;
  r.Reset();
  auto f = MakeFrame(42.0f);
  r.PushFrame(f.data());
  const float* got = r.FrameFromNewest(0);
  ASSERT_NE(got, nullptr);
  for (size_t i = 0; i < FeatureRingBuffer::kFrameWidth; ++i) {
    EXPECT_FLOAT_EQ(got[i], f[i]) << "at " << i;
  }
}

TEST(FeatureRingBuffer, ResetClearsState) {
  FeatureRingBuffer r;
  r.Reset();
  for (int i = 0; i < KWS_NUM_FRAMES; ++i) {
    auto f = MakeFrame(static_cast<float>(i));
    r.PushFrame(f.data());
  }
  ASSERT_TRUE(r.WindowReady());
  r.Reset();
  EXPECT_FALSE(r.WindowReady());
  EXPECT_EQ(r.count(), 0u);
  EXPECT_EQ(r.total_frames(), 0u);
}

TEST(FeatureRingBuffer, StorageIsStatic) {
  EXPECT_GE(sizeof(FeatureRingBuffer),
            FeatureRingBuffer::kCapacityFrames * FeatureRingBuffer::kFrameWidth * sizeof(float));
}
