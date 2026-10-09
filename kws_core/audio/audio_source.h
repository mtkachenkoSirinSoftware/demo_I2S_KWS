// SPDX-License-Identifier: Apache-2.0
// Acquisition backend seam. Selected at compile time — no vtable, no heap.
// Switching UART / MIC / CLIP must not touch dsp/, model/ or postprocessing/.
//
// kws_core itself does not include this header (FlashClipPlayer uses
// audio_status.h). Example mains and board HAL include it with exactly one
// KWS_AUDIO_SOURCE_* define.
#ifndef AUDIO_SOURCE_H_
#define AUDIO_SOURCE_H_

#include <stdbool.h>
#include <stdint.h>

#include "audio/audio_status.h"
#include "config/kws_config.h"

#if defined(KWS_AUDIO_SOURCE_UART) + defined(KWS_AUDIO_SOURCE_MIC) + \
        defined(KWS_AUDIO_SOURCE_CLIP) !=                            \
    1
#error "define exactly one of KWS_AUDIO_SOURCE_UART / KWS_AUDIO_SOURCE_MIC / KWS_AUDIO_SOURCE_CLIP"
#endif

#ifdef __cplusplus
extern "C" {
#endif

bool AudioSourceInit(void);
bool AudioSourceStart(void);
bool AudioSourceStop(void);

bool AudioSourceBlockReady(void);
const AudioSample* AudioSourceAcquireBlock(uint32_t* out_samples);
void AudioSourceReleaseBlock(void);
const AudioSourceStatus* AudioSourceGetStatus(void);

// UART START/END/RESET and flash-clip begin/end. Microphone always returns false.
bool AudioSourceTakeStreamBegin(void);
bool AudioSourceTakeStreamEnd(void);
bool AudioSourceTakeReset(void);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // AUDIO_SOURCE_H_
