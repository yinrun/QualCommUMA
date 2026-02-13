cmake_minimum_required(VERSION 3.18)
project(qnn_htp_add_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED QNN_SDK_ROOT)
  message(FATAL_ERROR "QNN_SDK_ROOT is not set")
endif()

add_executable(qnn_htp_add_demo src/main.cpp)
target_include_directories(qnn_htp_add_demo PRIVATE
  "${QNN_SDK_ROOT}/include"
  "${QNN_SDK_ROOT}/include/QNN"
)
target_link_libraries(qnn_htp_add_demo PRIVATE dl)
