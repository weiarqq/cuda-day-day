cmake_minimum_required(VERSION 3.18)
project(cuda-day-day VERSION 0.1.0 LANGUAGES C CXX CUDA)


# 查询 cudatoolkit
find_package(CUDAToolkit REQUIRED)
if (CMAKE_CUDA_COMPILER)
    message(STATUS "nvcc path : ${CMAKE_CUDA_COMPILER}")
else ()
    message(WARNING "nvcc not found. Please check CUDA is installed correctly!")
endif ()

# 指定C++标准（CUDA也继承此标准）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)  # 显式指定CUDA标准
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA架构（推荐用target属性，而非全局变量）
set(CMAKE_CUDA_ARCHITECTURES 89)

# 调试/发布模式的CUDA编译选项
option(ENABLE_CUDA_DEBUG "Enable CUDA debug (cuda-gdb)" OFF)
if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")        # enable cuda-gdb (may significantly affect performance on some targets)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()

# 构建可执行文件
add_executable(checkBestGPU checkBestGPU.cu)
add_executable(checkDeviceInfor checkDeviceInfor.cu)
add_executable(checkDimension checkDimension.cu)
add_executable(checkThreadIndex checkThreadIndex.cu)
add_executable(defineGridBlock defineGridBlock.cu)
add_executable(sumArraysOnGPU_small_case sumArraysOnGPU_small_case.cu)
add_executable(sumArraysOnGPU_timer sumArraysOnGPU_timer.cu)
add_executable(sumArraysOnHost sumArraysOnHost.c)
add_executable(sumMatrixOnGPU-2D-grid-1D-block sumMatrixOnGPU-2D-grid-1D-block.cu)
add_executable(sumMatrixOnGPU-1D-grid-1D-block sumMatrixOnGPU-1D-grid-1D-block.cu)
add_executable(sumMatrixOnGPU-2D-grid-2D-block sumMatrixOnGPU-2D-grid-2D-block.cu)