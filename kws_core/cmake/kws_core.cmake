# SPDX-License-Identifier: Apache-2.0
# Source lists for kws_core. Included by:
#   kws_core/CMakeLists.txt                     (standalone host tests)
#   kws_clip / kws_uart / kws_pdm CMakeLists    (NSX images)
#
# UART FIFO-poll HAL + framing live in kws_uart/, not here. kws_core stays
# HAL-free (FlashClipPlayer only).
if(NOT DEFINED KWS_CORE_ROOT)
  get_filename_component(KWS_CORE_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
endif()

set(KWS_CORE_SOURCES
  ${KWS_CORE_ROOT}/app/kws_app.cc
  ${KWS_CORE_ROOT}/audio/audio_ring_buffer.cc
  ${KWS_CORE_ROOT}/audio/flash_clip_player.c
  ${KWS_CORE_ROOT}/dsp/feature_ring_buffer.cc
  ${KWS_CORE_ROOT}/dsp/streaming_frontend.cc
  ${KWS_CORE_ROOT}/model/model_runner.cc
  ${KWS_CORE_ROOT}/model/quantizer.cc
  ${KWS_CORE_ROOT}/model/backends/runner_host_stub.cc
  ${KWS_CORE_ROOT}/model/backends/runner_helia_rt.cc
  ${KWS_CORE_ROOT}/postprocessing/recognizer.cc
)

# Host-only: cycle counter returns 0 (not an Apollo510 measurement).
set(KWS_CORE_HOST_SOURCES
  ${KWS_CORE_SOURCES}
  ${KWS_CORE_ROOT}/profiling/kws_profile_host.cc
)

set(KWS_CORE_TEST_SOURCES
  ${KWS_CORE_ROOT}/tests/test_audio_ring_buffer.cc
  ${KWS_CORE_ROOT}/tests/test_feature_ring_buffer.cc
  ${KWS_CORE_ROOT}/tests/test_streaming_frontend.cc
  ${KWS_CORE_ROOT}/tests/test_quantizer.cc
  ${KWS_CORE_ROOT}/tests/test_recognizer.cc
  ${KWS_CORE_ROOT}/tests/test_golden_frontend.cc
  ${KWS_CORE_ROOT}/tests/test_kws_app.cc
  ${KWS_CORE_ROOT}/tests/test_flash_clip.cc
)

set(KWS_CORE_GOLDEN_DIR "${KWS_CORE_ROOT}/tests/golden")
