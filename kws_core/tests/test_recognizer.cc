// SPDX-License-Identifier: Apache-2.0
#include "postprocessing/recognizer.h"

#include <gtest/gtest.h>

#include <vector>

using kws::KeywordEvent;
using kws::Recognizer;

namespace {

// One-hot-ish score vector with `p` on `label` and the rest spread evenly.
std::vector<float> Scores(int label, float p) {
  std::vector<float> s(KWS_NUM_CLASSES, (1.0f - p) / (KWS_NUM_CLASSES - 1));
  s[label] = p;
  return s;
}

constexpr int kYes = 9;        // kLabels[9]
constexpr int kSilence = 10;
constexpr int kUnknown = 11;
constexpr uint32_t kStrideMs = 100;  // KWS_INFERENCE_STRIDE_FRAMES * 20 ms

}  // namespace

TEST(Recognizer, SilentStreamNeverFires) {
  Recognizer r;
  r.Init();
  for (int i = 0; i < 200; ++i) {
    auto s = Scores(kSilence, 0.99f);
    EXPECT_FALSE(r.Update(s.data(), i * kStrideMs).fired);
  }
  EXPECT_EQ(r.events_fired(), 0u);
}

// _silence_ / _unknown_ are handled, not reported as keywords.
TEST(Recognizer, UnknownNeverFiresEvenWhenConfident) {
  Recognizer r;
  r.Init();
  for (int i = 0; i < 50; ++i) {
    auto s = Scores(kUnknown, 1.0f);
    EXPECT_FALSE(r.Update(s.data(), i * kStrideMs).fired);
  }
  EXPECT_EQ(r.events_fired(), 0u);
}

TEST(Recognizer, LowConfidenceDoesNotFire) {
  Recognizer r;
  r.Init();
  for (int i = 0; i < 50; ++i) {
    auto s = Scores(kYes, 0.40f);  // below KWS_DETECTION_THRESHOLD
    EXPECT_FALSE(r.Update(s.data(), i * kStrideMs).fired);
  }
  EXPECT_EQ(r.events_fired(), 0u);
}

TEST(Recognizer, RequiresConsistentDetectionsBeforeFiring) {
  Recognizer r;
  r.Init();
  r.set_min_consistent(3);
  int fired_at = -1;
  for (int i = 0; i < 20; ++i) {
    auto s = Scores(kYes, 1.0f);
    if (r.Update(s.data(), i * kStrideMs).fired) { fired_at = i; break; }
  }
  // Smoothing needs to lift the average over threshold, then the consistency
  // counter needs min_consistent hits; it must not fire on the very first one.
  ASSERT_GE(fired_at, 1);
}

// One spoken keyword -> one event.
TEST(Recognizer, OneKeywordProducesExactlyOneEvent) {
  Recognizer r;
  r.Init();
  uint32_t t = 0;
  // ~600 ms of the keyword dominating overlapping windows.
  for (int i = 0; i < 6; ++i, t += kStrideMs) {
    auto s = Scores(kYes, 0.98f);
    r.Update(s.data(), t);
  }
  // then back to silence
  for (int i = 0; i < 10; ++i, t += kStrideMs) {
    auto s = Scores(kSilence, 0.98f);
    r.Update(s.data(), t);
  }
  EXPECT_EQ(r.events_fired(), 1u);
}

// T0.1: a stuck timestamp makes (now - last) == 0, so debounce never expires.
TEST(Recognizer, FrozenNowMsSuppressesForeverAfterFirst) {
  Recognizer r;
  r.Init();
  for (int i = 0; i < 12; ++i) {
    auto s = Scores(kYes, 1.0f);
    r.Update(s.data(), 0);
  }
  EXPECT_EQ(r.events_fired(), 1u);
}

TEST(Recognizer, SuppressionBlocksRepeatsWithinWindow) {
  Recognizer r;
  r.Init();
  r.set_suppression_ms(750);
  uint32_t t = 0;
  for (int i = 0; i < 40; ++i, t += kStrideMs) {
    auto s = Scores(kYes, 1.0f);
    r.Update(s.data(), t);
  }
  // 40 * 100 ms = 4000 ms of continuous keyword. With 750 ms suppression the
  // theoretical maximum is ~6 events; a missing debounce would give ~38.
  EXPECT_LE(r.events_fired(), 6u);
  EXPECT_GE(r.events_fired(), 1u);
}

TEST(Recognizer, TwoSeparatedKeywordsProduceTwoEvents) {
  Recognizer r;
  r.Init();
  uint32_t t = 0;
  auto say = [&](int label, int n) {
    for (int i = 0; i < n; ++i, t += kStrideMs) {
      auto s = Scores(label, 0.98f);
      r.Update(s.data(), t);
    }
  };
  say(kYes, 6);
  say(kSilence, 20);  // 2 s gap, well past suppression
  say(kYes, 6);
  EXPECT_EQ(r.events_fired(), 2u);
}

TEST(Recognizer, EventCarriesLabelScoreAndTimestamp) {
  Recognizer r;
  r.Init();
  uint32_t t = 5000;
  KeywordEvent last;
  for (int i = 0; i < 8; ++i, t += kStrideMs) {
    auto s = Scores(kYes, 0.99f);
    KeywordEvent e = r.Update(s.data(), t);
    if (e.fired) { last = e; break; }
  }
  ASSERT_TRUE(last.fired);
  EXPECT_EQ(last.label, kYes);
  EXPECT_GT(last.score, KWS_DETECTION_THRESHOLD);
  EXPECT_GE(last.timestamp_ms, 5000u);
}

// Smoothing must reject a single spurious spike between confident silences.
TEST(Recognizer, SingleSpikeIsSmoothedAway) {
  Recognizer r;
  r.Init();
  uint32_t t = 0;
  for (int i = 0; i < 5; ++i, t += kStrideMs) {
    auto s = Scores(kSilence, 0.99f);
    r.Update(s.data(), t);
  }
  auto spike = Scores(kYes, 1.0f);
  EXPECT_FALSE(r.Update(spike.data(), t).fired);
  t += kStrideMs;
  for (int i = 0; i < 5; ++i, t += kStrideMs) {
    auto s = Scores(kSilence, 0.99f);
    r.Update(s.data(), t);
  }
  EXPECT_EQ(r.events_fired(), 0u);
}

TEST(Recognizer, SwitchingKeywordResetsConsistency) {
  Recognizer r;
  r.Init();
  r.set_min_consistent(3);
  uint32_t t = 0;
  // Alternate two keywords so neither ever reaches 3 consecutive hits.
  for (int i = 0; i < 30; ++i, t += kStrideMs) {
    auto s = Scores((i % 2 == 0) ? 0 : 1, 0.99f);
    r.Update(s.data(), t);
  }
  EXPECT_EQ(r.events_fired(), 0u);
}

TEST(Recognizer, ResetClearsEverything) {
  Recognizer r;
  r.Init();
  uint32_t t = 0;
  for (int i = 0; i < 10; ++i, t += kStrideMs) {
    auto s = Scores(kYes, 0.99f);
    r.Update(s.data(), t);
  }
  ASSERT_GT(r.events_fired(), 0u);
  r.Reset();
  EXPECT_EQ(r.events_fired(), 0u);
}

TEST(Recognizer, NullScoresAreIgnored) {
  Recognizer r;
  r.Init();
  EXPECT_FALSE(r.Update(nullptr, 0).fired);
}
