cmake_minimum_required(VERSION 3.18)
project(cuda-day-day VERSION 0.1.0 LANGUAGES C CXX)



enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)
if (CMAKE_CUDA_COMPILER)
    message(STATUS "nvcc path : ${CMAKE_CUDA_COMPILER}")
else ()
    message(WARNING "nvcc not found. Please check CUDA is installed correctly!")
endif ()


set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90 100 110 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")        # enable cuda-gdb (may significantly affect performance on some targets)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

# set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_ARCHITECTURES "50-virtual;61-virtual;70-virtual;75-virtual;80-virtual;86-real;89-real")
# file(GLOB CUDA_HEAD_FILES include/*.cuh)
file(GLOB CUDA_SOURCE_FILES src/*.cu )
# list(APPEND CUDA_SOURCE_FILES "main.cpp")
message(STATUS "Src files: ${CUDA_SOURCE_FILES}")
message(STATUS "Include  files: ${CUDA_HEAD_FILES}")




add_executable(cuda-day-day ${CUDA_SOURCE_FILES})

