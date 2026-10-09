// SPDX-License-Identifier: Apache-2.0
// Fixed-size circular PCM buffer. No malloc/new, ever.
#ifndef AUDIO_RING_BUFFER_H_
#define AUDIO_RING_BUFFER_H_

#include <stddef.h>
#include <stdint.h>

#include "config/kws_config.h"

namespace kws {

// Single-producer (audio block consumer in the main loop) / single-consumer
// (frontend) ring. Writes never block and never allocate; when the producer
// outruns the consumer the oldest samples are dropped and `overruns()` counts
// it — silently losing audio is not acceptable.
class AudioRingBuffer {
 public:
  AudioRingBuffer() = default;

  void Reset();

  // Appends `count` samples. Returns false if the push overwrote unread data
  // (and bumps the overrun counter); the samples are still stored.
  bool Push(const AudioSample* samples, size_t count);

  // Number of samples written but not yet consumed via Advance().
  size_t Available() const { return available_; }

  static constexpr size_t Capacity() { return KWS_PCM_RING_SAMPLES; }

  // Copies the most recent `count` samples ending at the write cursor into
  // `out`. Returns false if fewer than `count` samples have ever been written.
  // This is the only copying read; it exists so the frontend can lift one
  // analysis frame without the ring exposing its wrap point.
  bool PeekNewest(AudioSample* out, size_t count) const;

  // Copies `count` samples starting `offset_from_oldest` samples after the
  // read cursor. Used by the frontend to walk frames in order.
  bool Read(size_t offset_from_oldest, AudioSample* out, size_t count) const;

  // Marks `count` samples consumed, freeing ring space.
  void Advance(size_t count);

  uint32_t overruns() const { return overruns_; }
  uint64_t total_written() const { return total_written_; }

 private:
  AudioSample buffer_[KWS_PCM_RING_SAMPLES] = {};
  size_t write_ = 0;      // next slot to write
  size_t read_ = 0;       // oldest unconsumed slot
  size_t available_ = 0;  // unconsumed sample count
  uint32_t overruns_ = 0;
  uint64_t total_written_ = 0;
};

}  // namespace kws
#endif  // AUDIO_RING_BUFFER_H_
