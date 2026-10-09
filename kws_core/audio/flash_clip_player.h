// SPDX-License-Identifier: Apache-2.0
// One-shot PCM player. Same hop size as DMA (KWS_AUDIO_BLOCK_SAMPLES).
// Host-testable: no HAL. Target AudioSource is a thin wrapper around this.
#ifndef FLASH_CLIP_PLAYER_H_
#define FLASH_CLIP_PLAYER_H_

#include <stdbool.h>
#include <stdint.h>

#include "audio/audio_status.h"
#include "config/kws_config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FlashClipPlayer {
  const AudioSample* pcm;
  uint32_t n_samples;
  uint32_t offset;
  bool started;
  bool begin_pending;
  bool end_pending;
  bool ended;
  AudioSample block[KWS_AUDIO_BLOCK_SAMPLES];
  AudioSourceStatus status;
} FlashClipPlayer;

bool FlashClipPlayerInit(FlashClipPlayer* p, const AudioSample* pcm, uint32_t n_samples);
bool FlashClipPlayerStart(FlashClipPlayer* p);
bool FlashClipPlayerStop(FlashClipPlayer* p);

bool FlashClipPlayerBlockReady(const FlashClipPlayer* p);
const AudioSample* FlashClipPlayerAcquireBlock(FlashClipPlayer* p, uint32_t* out_samples);
void FlashClipPlayerReleaseBlock(FlashClipPlayer* p);
const AudioSourceStatus* FlashClipPlayerStatus(const FlashClipPlayer* p);

bool FlashClipPlayerTakeStreamBegin(FlashClipPlayer* p);
bool FlashClipPlayerTakeStreamEnd(FlashClipPlayer* p);
bool FlashClipPlayerTakeReset(FlashClipPlayer* p);

#ifdef __cplusplus
}
#endif
#endif  // FLASH_CLIP_PLAYER_H_
