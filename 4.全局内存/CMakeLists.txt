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

function(add_cudadevrt target_name)
    # 使用${target_name}引用函数参数（变量），而非字面量

    # 启用分离编译（替代手动设置 -rdc=true，CMake会自动处理设备链接）
    set_target_properties(${target_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON  # 关键：启用分离编译，自动添加-rdc=true和设备链接步骤
        CUDA_RESOLVE_DEVICE_SYMBOLS ON # 确保设备符号解析（分离编译必需）
    )

    # 链接CUDA库
    target_include_directories(${target_name} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_directories(${target_name} PRIVATE ${CUDAToolkit_LIBRARY_DIR})
    # 直接指定库名称（CUDAToolkit提供的变量）
    target_link_libraries(${target_name} PRIVATE
        ${CUDAToolkit_CUDA_LIBRARY}    # 替代CUDAToolkit::CUDA
        ${CUDAToolkit_cudadevrt_LIBRARY} # 替代CUDAToolkit::cudadevrt
    )

endfunction()


# 调试/发布模式的CUDA编译选项
option(ENABLE_CUDA_DEBUG "Enable CUDA debug (cuda-gdb)" OFF)
if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")        # enable cuda-gdb (may significantly affect performance on some targets)
else()
    message("==================================")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -Xptxas -dlcm=cg") # add line information to all builds for debug tools (exclusive to -G option)
endif()


set(COMMON_FILES common.cpp)

# add_executable(memTransfer memTransfer.cu ${COMMON_FILES})
# add_executable(pinMemTransfer pinMemTransfer.cu ${COMMON_FILES})
# add_executable(sumArrayZerocopy sumArrayZerocopy.cu ${COMMON_FILES})
# add_executable(sumArrayZerocopyUVA sumArrayZerocopyUVA.cu ${COMMON_FILES})
# add_executable(readSegment readSegment.cu ${COMMON_FILES})
# add_executable(writeSegment writeSegment.cu ${COMMON_FILES})
# add_executable(simpleMethAoS simpleMethAoS.cu ${COMMON_FILES})
# add_executable(simpleMethSoA simpleMethSoA.cu ${COMMON_FILES})
# add_executable(readSegmentUnroll readSegmentUnroll.cu ${COMMON_FILES})
# add_executable(transpose transpose.cu ${COMMON_FILES})
# add_executable(sumMatrixGPUManaged sumMatrixGPUManaged.cu ${COMMON_FILES})
# add_executable(sumMatrixGPUManual sumMatrixGPUManual.cu ${COMMON_FILES})
add_executable(globalVariable globalVariable.cu ${COMMON_FILES})
