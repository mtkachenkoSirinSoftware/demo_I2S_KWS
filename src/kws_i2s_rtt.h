// SPDX-License-Identifier: Apache-2.0
// Optional RTT channel-1 dump of the int16 hop handed to kws_core.
// Labels stay on SWO.
#ifndef KWS_I2S_RTT_H_
#define KWS_I2S_RTT_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { kKwsI2sRttPcmChannel = 1 };

void KwsI2sRttInit(void);
void KwsI2sRttWriteHop(const int16_t* pcm, uint32_t n_samples);

#ifdef __cplusplus
}
#endif
#endif  // KWS_I2S_RTT_H_
