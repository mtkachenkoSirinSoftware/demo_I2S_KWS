# SPDX-License-Identifier: Apache-2.0
# Standalone host build of kws_core (no UART HAL, no NSX, no EVB).
#
#   cmake -S . -B build
#   cmake --build build -j
#   ctest --test-dir build --output-on-failure
cmake_minimum_required(VERSION 3.16)
project(kws_core C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wno-unused-parameter)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/kws_core.cmake)

add_library(kws_core ${KWS_CORE_HOST_SOURCES})
target_include_directories(kws_core PUBLIC ${KWS_CORE_ROOT})

include(FetchContent)
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_executable(kws_core_tests ${KWS_CORE_TEST_SOURCES})
target_link_libraries(kws_core_tests PRIVATE kws_core GTest::gtest_main)
target_compile_definitions(kws_core_tests PRIVATE
  KWS_GOLDEN_DIR="${KWS_CORE_GOLDEN_DIR}")

include(GoogleTest)
gtest_discover_tests(kws_core_tests)
