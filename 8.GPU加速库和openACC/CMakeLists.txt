cmake_minimum_required(VERSION 3.18)
project(cuda-day-day VERSION 0.1.0 LANGUAGES C CXX CUDA)


# 查询 cudatoolkit
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

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

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler ${OpenMP_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# 调试/发布模式的CUDA编译选项
option(ENABLE_CUDA_DEBUG "Enable CUDA debug (cuda-gdb)" OFF)
if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")        # enable cuda-gdb (may significantly affect performance on some targets)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -Xptxas -dlcm=cg") # add line information to all builds for debug tools (exclusive to -G option)
    message("CUDA compiler:" ${CMAKE_CUDA_FLAGS})
    message("CPP compiler:" ${CMAKE_CXX_FLAGS})

endif()

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

add_executable(cublas cublas.cu)
target_link_libraries(cublas PRIVATE ${CUBLAS_LIB})
set_target_properties(cublas PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_executable(cuda-openacc cuda-openacc.cu)
target_link_libraries(cuda-openacc PRIVATE ${CUBLAS_LIB} ${CURAND_LIB})
set_target_properties(cuda-openacc PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_executable(cufft-multi cufft-multi.cu)
target_link_libraries(cufft-multi PRIVATE ${CUFFT_LIB})
set_target_properties(cufft-multi PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_executable(cufft cufft.cu)
target_link_libraries(cufft PRIVATE ${CUFFT_LIB})
set_target_properties(cufft PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_executable(cusparse cusparse.cu)
target_link_libraries(cusparse PRIVATE ${CUSPARSE_LIB})
set_target_properties(cusparse PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_executable(rand-kernel rand-kernel.cu)
target_link_libraries(rand-kernel PRIVATE ${CURAND_LIB})
set_target_properties(rand-kernel PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_executable(replace-rand-streams replace-rand-streams.cu)
target_link_libraries(replace-rand-streams PRIVATE ${CURAND_LIB})
set_target_properties(replace-rand-streams PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_executable(replace-rand replace-rand.cu)
target_link_libraries(replace-rand PRIVATE ${CURAND_LIB})
set_target_properties(replace-rand PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_executable(simple-data simple-data.c  ${COMMON_FILES})
add_executable(simple-kernels simple-kernels.c  ${COMMON_FILES})
add_executable(simple-parallel simple-parallel.c  ${COMMON_FILES})

