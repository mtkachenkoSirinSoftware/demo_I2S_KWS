// SPDX-License-Identifier: Apache-2.0
// The embedded model flatbuffer.
//
// The definition lives in model/generated/kws_model_data.cc, which is written
// by tools/embed_model.py from a .tflite and is NOT committed — it is a
// mechanical restatement of a file that already exists elsewhere in the repo,
// and committing 100 kB of hex would make every re-export a reviewable diff for
// no information gain. Regenerate it with:
//
//   python3 tools/embed_model.py <path>.tflite
//
// The same run writes model/generated/kws_model_contract.h (committed, because
// the scales, zero points and SHA256 in it *are* provenance) and
// kws_model_manifest.json. config/kws_config.h *consumes* those scales when
// the contract header is present — Class B models do not need hand edits.
#ifndef KWS_MODEL_DATA_H_
#define KWS_MODEL_DATA_H_

extern const unsigned char g_kws_model_data[];
extern const unsigned int g_kws_model_data_len;

#endif  // KWS_MODEL_DATA_H_
