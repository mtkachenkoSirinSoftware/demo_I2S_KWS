# SPDX-License-Identifier: Apache-2.0
#
# kws_i2s — always-on KWS from an I2S MEMS microphone through the local copy
# of kws_core, labels on SWO.
#
# After `nsx configure --app-dir .` this file is the firmware app.
# Before that, `cmake -S .` registers the host tests only.
cmake_minimum_required(VERSION 3.16)

set(KWS_I2S_ROOT "${CMAKE_CURRENT_LIST_DIR}")
set(KWS_CORE_ROOT "${KWS_I2S_ROOT}/kws_core")
set(NSX_BOOTSTRAP "${KWS_I2S_ROOT}/cmake/nsx/nsx_app_bootstrap.cmake")

if(NOT EXISTS "${NSX_BOOTSTRAP}")
  project(kws_i2s_host C CXX)
  enable_testing()
  add_subdirectory(tests)
  return()
endif()

project(kws_i2s LANGUAGES C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(NSX_BOARD "apollo510_evb" CACHE STRING "Target board")
set(NSX_HELIA_RT_BACKEND "helia" CACHE STRING "heliaRT kernels: helia | cmsis_nn | reference")
set(NSX_CMSIS_NN_ENABLE_F32 ON CACHE BOOL "" FORCE)
set(NSX_CMSIS_NN_ENABLE_F16 ON CACHE BOOL "" FORCE)
set(HELIA_RT_VARIANT "release-with-logs" CACHE STRING "debug | release-with-logs | release")

include(${KWS_I2S_ROOT}/cmake/nsx/modules.cmake)
include(${NSX_BOOTSTRAP})

nsx_bootstrap_app(
    APP_ROOT "${KWS_I2S_ROOT}"
    BOARD "${NSX_BOARD}"
    MODULES ${NSX_APP_MODULES}
)

include(${KWS_CORE_ROOT}/cmake/kws_core.cmake)

set(_model_cc "${KWS_I2S_ROOT}/model/generated/kws_model_data.cc")
set(_embed_model "${KWS_I2S_ROOT}/tools/embed_model.py")
set(_tflite "${KWS_I2S_ROOT}/assets/depgraph_r060_kd_int8.tflite")

if(NOT EXISTS "${_model_cc}")
  execute_process(
    COMMAND python3 "${_embed_model}" "${_tflite}" --out-dir "${KWS_I2S_ROOT}/model/generated"
    RESULT_VARIABLE _rv)
  if(NOT _rv EQUAL 0)
    message(FATAL_ERROR "embed_model.py failed (${_rv}). Need ${_tflite}.")
  endif()
endif()

option(KWS_I2S_SLEEP "Sleep the core between hops (always-on loop)" ON)
option(KWS_I2S_INFER_LINES "Print one `inf` line per inference (10/s) on SWO" ON)
option(KWS_I2S_RTT_PCM "Dump the int16 hop on RTT channel 1 (labels stay on SWO)" OFF)
set(KWS_I2S_PCM_SHIFT "8" CACHE STRING "Right shift from the 24-bit sample to int16")
set(KWS_I2S_HPF_HZ "60" CACHE STRING "High-pass cutoff in Hz (0 disables)")
set(KWS_I2S_STARTUP_HOPS "100" CACHE STRING "20 ms hops muted after start (microphone settling)")

set(_i2s_src src/main.cc src/i2s_audio_source.c src/kws_profile_target.c)
if(KWS_I2S_RTT_PCM)
  list(APPEND _i2s_src src/kws_i2s_rtt.c rtt/RTT/SEGGER_RTT.c)
endif()

add_executable(kws_i2s
    ${_i2s_src}
    ${KWS_CORE_SOURCES}
    ${_model_cc}
)

target_include_directories(kws_i2s PRIVATE
    ${KWS_I2S_ROOT}
    ${KWS_I2S_ROOT}/src
    ${KWS_CORE_ROOT}
)
if(KWS_I2S_RTT_PCM)
  target_include_directories(kws_i2s PRIVATE
      ${KWS_I2S_ROOT}/rtt/RTT
      ${KWS_I2S_ROOT}/rtt/Config
  )
endif()

target_compile_definitions(kws_i2s PRIVATE
    KWS_TARGET_APOLLO510=1
    KWS_USE_NSX_MEM=1
    KWS_RUNTIME_HELIA_RT=1
    KWS_AUDIO_SOURCE_MIC=1
    KWS_I2S_PCM_SHIFT=${KWS_I2S_PCM_SHIFT}u
    KWS_I2S_HPF_HZ=${KWS_I2S_HPF_HZ}u
    KWS_I2S_STARTUP_HOPS=${KWS_I2S_STARTUP_HOPS}u
)
if(KWS_I2S_SLEEP)
  target_compile_definitions(kws_i2s PRIVATE KWS_I2S_SLEEP=1)
endif()
if(KWS_I2S_INFER_LINES)
  target_compile_definitions(kws_i2s PRIVATE KWS_I2S_INFER_LINES=1)
endif()
if(KWS_I2S_RTT_PCM)
  target_compile_definitions(kws_i2s PRIVATE KWS_I2S_RTT_PCM=1)
endif()

target_link_libraries(kws_i2s PRIVATE
    nsx::board
    nsx::runtime_core
    nsx::power
    nsx::helia_rt
)

nsx_finalize_app(kws_i2s)
