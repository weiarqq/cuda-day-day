cmake_minimum_required(VERSION 3.18)
project(cuda-day-day VERSION 0.1.0 LANGUAGES C CXX CUDA)

find_package(OpenMP REQUIRED)
find_library(CUBLAS_LIB cublas PATHS $ENV{CUDA_HOME}/lib64 /usr/local/cuda/lib64)
find_library(CURAND_LIB curand PATHS $ENV{CUDA_HOME}/lib64 /usr/local/cuda/lib64)
find_library(CUFFT_LIB cufft PATHS $ENV{CUDA_HOME}/lib64 /usr/local/cuda/lib64)
find_library(CUSPARSE_LIB cusparse PATHS $ENV{CUDA_HOME}/lib64 /usr/local/cuda/lib64)
# 验证库是否找到
if(NOT CUBLAS_LIB)
    message(FATAL_ERROR "cublas library not found! Set CUDA_HOME environment variable.")
endif()
if(NOT CURAND_LIB)
    message(FATAL_ERROR "curand library not found! Set CUDA_HOME environment variable.")
endif()
if(NOT CUFFT_LIB)
    message(FATAL_ERROR "cufft library not found! Set CUDA_HOME environment variable.")
endif()
if(NOT CUSPARSE_LIB)
    message(FATAL_ERROR "cusparse library not found! Set CUDA_HOME environment variable.")
endif()
set(COMMON_FILES common.cpp)
# 验证库是否找到
add_executable(v1 v1.cu)
target_link_libraries(v1 PRIVATE ${CUBLAS_LIB})
set_target_properties(v1 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)