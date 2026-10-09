# SPDX-License-Identifier: Apache-2.0
# Host tests for demo_I2S (no EVB): I2S word -> PCM16 conditioning and the
# dashboard's SWO parser. Standalone (`cmake -S tests`) or via the top-level
# CMakeLists before `nsx configure` has run.
cmake_minimum_required(VERSION 3.16)
project(kws_i2s_host_check C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(KWS_I2S_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")
set(KWS_CORE_ROOT "${KWS_I2S_ROOT}/kws_core")

add_executable(kws_i2s_check check_i2s.cc)
target_include_directories(kws_i2s_check PRIVATE ${KWS_I2S_ROOT} ${KWS_I2S_ROOT}/src ${KWS_CORE_ROOT})
target_compile_options(kws_i2s_check PRIVATE -Wall -Wextra)

enable_testing()
add_test(NAME kws_i2s_check COMMAND kws_i2s_check)
add_test(NAME kws_dashboard_parser
         COMMAND python3 "${CMAKE_CURRENT_LIST_DIR}/test_dashboard_parser.py")
