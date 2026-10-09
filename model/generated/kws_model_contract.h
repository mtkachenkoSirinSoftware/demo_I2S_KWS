// SPDX-License-Identifier: Apache-2.0
// GENERATED FILE — do not edit.
// Produced by tools/embed_model.py from:
//   ../assets/depgraph_r060_kd_int8.tflite
//   SHA256 ae08012b5a5dd1fd59673bba3d4b1d1c43d7cd1f410537824d7eb6c2edb40fb7
//   18256 bytes, TFLite schema v3
//
// Everything below was read out of the flatbuffer, never assumed. Include it
// from config/kws_config.h — that header *consumes* these scales/zero points
// so a Class B re-quantised model does not require hand-editing literals.
//
// Graph, in execution order:
//   PAD
//   CONV_2D
//   DEPTHWISE_CONV_2D
//   CONV_2D
//   DEPTHWISE_CONV_2D
//   CONV_2D
//   DEPTHWISE_CONV_2D
//   CONV_2D
//   DEPTHWISE_CONV_2D
//   CONV_2D
//   AVERAGE_POOL_2D
//   RESHAPE
//   FULLY_CONNECTED
#ifndef KWS_MODEL_CONTRACT_H_
#define KWS_MODEL_CONTRACT_H_

#define KWS_MODEL_SHA256 "ae08012b5a5dd1fd59673bba3d4b1d1c43d7cd1f410537824d7eb6c2edb40fb7"
#define KWS_MODEL_BYTES  18256
#define KWS_MODEL_SCHEMA_VERSION 3
#define KWS_MODEL_NUM_OPERATORS  13
#define KWS_MODEL_NUM_TENSORS    36
// Sum of every constant buffer: the flash cost of the weights alone.
#define KWS_MODEL_WEIGHT_BYTES   5891

// Distinct builtin ops -> the MicroMutableOpResolver<N> size the runner needs.
#define KWS_MODEL_NUM_OPS 6

// --- input tensor: serving_default_input_1:0 ---
#define KWS_MODEL_INPUT_ELEMENTS   490
#define KWS_MODEL_INPUT_TYPE_INT8 1
#define KWS_MODEL_INPUT_SCALE      0.5899888277053833f
#define KWS_MODEL_INPUT_ZERO_POINT 81

// --- output tensor: StatefulPartitionedCall:0 ---
#define KWS_MODEL_OUTPUT_ELEMENTS   12
#define KWS_MODEL_OUTPUT_TYPE_INT8 1
#define KWS_MODEL_OUTPUT_SCALE      0.20843878388404846f
#define KWS_MODEL_OUTPUT_ZERO_POINT 42

// Registration list for tflite::MicroMutableOpResolver<KWS_MODEL_NUM_OPS>.
// Method names were resolved against heliaRT's own
// tensorflow/lite/micro/micro_mutable_op_resolver.h — an op with no Add*
// method fails generation rather than reaching AllocateTensors() as a
// mis-read "arena too small".
#define KWS_MODEL_REGISTER_OPS(r) \
    do { \
    r.AddAveragePool2D(); \
    r.AddConv2D(); \
    r.AddDepthwiseConv2D(); \
    r.AddFullyConnected(); \
    r.AddPad(); \
    r.AddReshape(); \
    } while (0)

#endif  // KWS_MODEL_CONTRACT_H_
