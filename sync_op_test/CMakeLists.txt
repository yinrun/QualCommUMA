cmake_minimum_required(VERSION 3.18)
project(sync_op_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED QNN_SDK_ROOT)
  message(FATAL_ERROR "QNN_SDK_ROOT is not set. Use -DQNN_SDK_ROOT=/path/to/qairt")
endif()

add_executable(sync_op_test
  src/main.cpp
)

target_include_directories(sync_op_test PRIVATE
  "${QNN_SDK_ROOT}/include"
  "${QNN_SDK_ROOT}/include/QNN"
)

target_link_libraries(sync_op_test PRIVATE dl)
