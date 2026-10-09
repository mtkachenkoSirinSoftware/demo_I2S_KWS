// SPDX-License-Identifier: Apache-2.0
// Acquisition counters and ping-pong ownership. HAL-free: FlashClipPlayer and
// host tests use this without selecting CLIP / UART / MIC.
#ifndef AUDIO_STATUS_H_
#define AUDIO_STATUS_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Ping-pong buffer ownership. A plain bool cannot express "DMA is filling B
// while the CPU still holds A".
typedef enum {
  kBufferDmaOwned = 0,
  kBufferReadyForCpu,
  kBufferCpuProcessing,
  kBufferFree,
} BufferState;

typedef struct {
  volatile uint32_t blocks_received;
  volatile uint32_t blocks_processed;
  volatile uint32_t overruns;
  volatile uint32_t uart_errors;
  volatile uint32_t dma_faults;
  volatile int32_t pcm_peak_abs;
  volatile int32_t pcm_dc;
  volatile BufferState state[2];
} AudioSourceStatus;

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // AUDIO_STATUS_H_
