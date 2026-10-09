// SPDX-License-Identifier: Apache-2.0
#include "audio/flash_clip_player.h"

#include <string.h>

static void StatusClear(AudioSourceStatus* s) {
  s->blocks_received = 0;
  s->blocks_processed = 0;
  s->overruns = 0;
  s->uart_errors = 0;
  s->dma_faults = 0;
  s->pcm_peak_abs = 0;
  s->pcm_dc = 0;
  s->state[0] = kBufferFree;
  s->state[1] = kBufferFree;
}

bool FlashClipPlayerInit(FlashClipPlayer* p, const AudioSample* pcm, uint32_t n_samples) {
  if (p == NULL || pcm == NULL || n_samples == 0) return false;
  p->pcm = pcm;
  p->n_samples = n_samples;
  p->offset = 0;
  p->started = false;
  p->begin_pending = false;
  p->end_pending = false;
  p->ended = false;
  memset(p->block, 0, sizeof(p->block));
  StatusClear(&p->status);
  return true;
}

bool FlashClipPlayerStart(FlashClipPlayer* p) {
  if (p == NULL || p->pcm == NULL) return false;
  p->offset = 0;
  p->started = true;
  p->begin_pending = true;
  p->end_pending = false;
  p->ended = false;
  StatusClear(&p->status);
  p->status.state[0] = kBufferReadyForCpu;
  return true;
}

bool FlashClipPlayerStop(FlashClipPlayer* p) {
  if (p == NULL) return false;
  p->started = false;
  p->ended = true;
  return true;
}

bool FlashClipPlayerBlockReady(const FlashClipPlayer* p) {
  if (p == NULL || !p->started || p->ended) return false;
  return p->offset < p->n_samples;
}

const AudioSample* FlashClipPlayerAcquireBlock(FlashClipPlayer* p, uint32_t* out_samples) {
  if (!FlashClipPlayerBlockReady(p)) return NULL;
  p->status.state[0] = kBufferCpuProcessing;

  uint32_t remain = p->n_samples - p->offset;
  uint32_t n = remain < KWS_AUDIO_BLOCK_SAMPLES ? remain : KWS_AUDIO_BLOCK_SAMPLES;
  memcpy(p->block, p->pcm + p->offset, n * sizeof(AudioSample));
  if (n < KWS_AUDIO_BLOCK_SAMPLES) {
    memset(p->block + n, 0, (KWS_AUDIO_BLOCK_SAMPLES - n) * sizeof(AudioSample));
  }
  p->offset += n;

  int32_t peak = 0;
  int64_t sum = 0;
  for (uint32_t i = 0; i < KWS_AUDIO_BLOCK_SAMPLES; ++i) {
    const int32_t a = p->block[i] < 0 ? -(int32_t)p->block[i] : (int32_t)p->block[i];
    if (a > peak) peak = a;
    sum += p->block[i];
  }
  p->status.pcm_peak_abs = peak;
  p->status.pcm_dc = (int32_t)(sum / (int64_t)KWS_AUDIO_BLOCK_SAMPLES);
  p->status.blocks_received += 1u;

  if (out_samples != NULL) *out_samples = KWS_AUDIO_BLOCK_SAMPLES;
  return p->block;
}

void FlashClipPlayerReleaseBlock(FlashClipPlayer* p) {
  if (p == NULL) return;
  p->status.state[0] = kBufferFree;
  p->status.blocks_processed += 1u;
  if (p->offset >= p->n_samples && !p->ended) {
    p->ended = true;
    p->end_pending = true;
  } else if (p->offset < p->n_samples) {
    p->status.state[0] = kBufferReadyForCpu;
  }
}

const AudioSourceStatus* FlashClipPlayerStatus(const FlashClipPlayer* p) {
  return p != NULL ? &p->status : NULL;
}

bool FlashClipPlayerTakeStreamBegin(FlashClipPlayer* p) {
  if (p == NULL || !p->begin_pending) return false;
  p->begin_pending = false;
  return true;
}

bool FlashClipPlayerTakeStreamEnd(FlashClipPlayer* p) {
  if (p == NULL || !p->end_pending) return false;
  p->end_pending = false;
  return true;
}

bool FlashClipPlayerTakeReset(FlashClipPlayer* p) {
  (void)p;
  return false;
}
