// SPDX-License-Identifier: Apache-2.0
#include "audio/audio_ring_buffer.h"

#include <gtest/gtest.h>

#include <vector>

using kws::AudioRingBuffer;
using kws::AudioSample;

namespace {

std::vector<AudioSample> Ramp(int n, int start = 0) {
  std::vector<AudioSample> v(n);
  for (int i = 0; i < n; ++i) v[i] = static_cast<AudioSample>(start + i);
  return v;
}

}  // namespace

TEST(AudioRingBuffer, StartsEmpty) {
  AudioRingBuffer r;
  r.Reset();
  EXPECT_EQ(r.Available(), 0u);
  EXPECT_EQ(r.overruns(), 0u);
  EXPECT_EQ(r.total_written(), 0u);
}

TEST(AudioRingBuffer, PushThenReadPreservesOrder) {
  AudioRingBuffer r;
  r.Reset();
  auto in = Ramp(320);
  ASSERT_TRUE(r.Push(in.data(), in.size()));
  EXPECT_EQ(r.Available(), 320u);

  std::vector<AudioSample> out(320);
  ASSERT_TRUE(r.Read(0, out.data(), out.size()));
  EXPECT_EQ(out, in);
}

TEST(AudioRingBuffer, ReadBeyondAvailableFails) {
  AudioRingBuffer r;
  r.Reset();
  auto in = Ramp(100);
  ASSERT_TRUE(r.Push(in.data(), in.size()));
  std::vector<AudioSample> out(101);
  EXPECT_FALSE(r.Read(0, out.data(), 101));
  EXPECT_FALSE(r.Read(50, out.data(), 51));
}

TEST(AudioRingBuffer, AdvanceFreesSpace) {
  AudioRingBuffer r;
  r.Reset();
  auto in = Ramp(500);
  ASSERT_TRUE(r.Push(in.data(), in.size()));
  r.Advance(200);
  EXPECT_EQ(r.Available(), 300u);

  std::vector<AudioSample> out(300);
  ASSERT_TRUE(r.Read(0, out.data(), out.size()));
  // After advancing 200, the oldest unconsumed sample is in[200].
  EXPECT_EQ(out.front(), in[200]);
  EXPECT_EQ(out.back(), in[499]);
}

// The whole point of a ring: writing past the end must wrap, not corrupt.
TEST(AudioRingBuffer, WrapsAroundCorrectly) {
  AudioRingBuffer r;
  r.Reset();
  const size_t cap = AudioRingBuffer::Capacity();
  // Fill to just under capacity, consume most of it, then push across the wrap.
  auto a = Ramp(static_cast<int>(cap) - 10);
  ASSERT_TRUE(r.Push(a.data(), a.size()));
  r.Advance(cap - 100);
  auto b = Ramp(50, 100000 % 32768);
  ASSERT_TRUE(r.Push(b.data(), b.size()));
  EXPECT_EQ(r.overruns(), 0u);
  EXPECT_EQ(r.Available(), 140u);

  std::vector<AudioSample> out(140);
  ASSERT_TRUE(r.Read(0, out.data(), out.size()));
  for (size_t i = 0; i < 90; ++i) EXPECT_EQ(out[i], a[cap - 100 + i]) << "at " << i;
  for (size_t i = 0; i < 50; ++i) EXPECT_EQ(out[90 + i], b[i]) << "at " << i;
}

// an always-on system that silently loses audio is unacceptable.
TEST(AudioRingBuffer, CountsOverrunWhenProducerOutrunsConsumer) {
  AudioRingBuffer r;
  r.Reset();
  const size_t cap = AudioRingBuffer::Capacity();
  auto full = Ramp(static_cast<int>(cap));
  ASSERT_TRUE(r.Push(full.data(), full.size()));
  EXPECT_EQ(r.overruns(), 0u);

  auto extra = Ramp(1, 7);
  EXPECT_FALSE(r.Push(extra.data(), 1));
  EXPECT_EQ(r.overruns(), 1u);
  EXPECT_EQ(r.Available(), cap);
}

TEST(AudioRingBuffer, PushLargerThanCapacityKeepsNewestTail) {
  AudioRingBuffer r;
  r.Reset();
  const size_t cap = AudioRingBuffer::Capacity();
  auto huge = Ramp(static_cast<int>(cap) + 137);
  EXPECT_FALSE(r.Push(huge.data(), huge.size()));
  EXPECT_EQ(r.overruns(), 1u);

  std::vector<AudioSample> out(cap);
  ASSERT_TRUE(r.Read(0, out.data(), cap));
  // The tail of the oversized push is what survives.
  EXPECT_EQ(out.back(), huge.back());
}

TEST(AudioRingBuffer, PeekNewestReturnsMostRecentSamples) {
  AudioRingBuffer r;
  r.Reset();
  auto in = Ramp(1000);
  ASSERT_TRUE(r.Push(in.data(), in.size()));

  std::vector<AudioSample> out(480);
  ASSERT_TRUE(r.PeekNewest(out.data(), out.size()));
  for (size_t i = 0; i < 480; ++i) EXPECT_EQ(out[i], in[1000 - 480 + i]);
}

TEST(AudioRingBuffer, PeekNewestFailsBeforeEnoughHistory) {
  AudioRingBuffer r;
  r.Reset();
  auto in = Ramp(100);
  ASSERT_TRUE(r.Push(in.data(), in.size()));
  std::vector<AudioSample> out(480);
  EXPECT_FALSE(r.PeekNewest(out.data(), 480));
}

TEST(AudioRingBuffer, PeekNewestSurvivesWrap) {
  AudioRingBuffer r;
  r.Reset();
  const size_t cap = AudioRingBuffer::Capacity();
  // Stream well past capacity in hop-sized blocks, consuming as we go, so the
  // write cursor wraps several times.
  AudioSample counter = 0;
  for (size_t block = 0; block < (cap / 320) * 3; ++block) {
    AudioSample buf[320];
    for (int i = 0; i < 320; ++i) buf[i] = counter++;
    r.Push(buf, 320);
    r.Advance(320);
  }
  std::vector<AudioSample> out(480);
  ASSERT_TRUE(r.PeekNewest(out.data(), out.size()));
  for (size_t i = 0; i < 480; ++i) {
    EXPECT_EQ(out[i], static_cast<AudioSample>(counter - 480 + i)) << "at " << i;
  }
}

TEST(AudioRingBuffer, NoAllocationSizedStatically) {
  // Static storage: the object must be big
  // enough to hold the ring itself, proving it is not a heap handle.
  EXPECT_GE(sizeof(AudioRingBuffer), AudioRingBuffer::Capacity() * sizeof(AudioSample));
}
