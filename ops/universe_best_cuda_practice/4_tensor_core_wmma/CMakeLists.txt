cmake_minimum_required(VERSION 3.20.0)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.8/bin/nvcc")

project(universe_cuda_best_practice VERSION 0.1.0 LANGUAGES CUDA CXX C)
set(CMAKE_CUDA_ARCHITECTURES 89)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


find_package(CUDAToolkit)
# add_subdirectory(1_cuda_reduce_study)
# add_subdirectory(2_cuda_sgemm_study)
# add_subdirectory(3_kernel_profiling_guide)
# add_subdirectory(4_tensor_core_wmma)
# add_subdirectory(5_mma_and_swizzle)
add_executable(hgemm_v1_wmma_m16n16k16_naive_kernel hgemm_v1_wmma_m16n16k16_naive_kernel.cu)
target_link_libraries(hgemm_v1_wmma_m16n16k16_naive_kernel PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(hgemm_v1_wmma_m16n16k16_naive_kernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
# target_compile_options(hgemm_wmma_m16n16k16_naive_kernel PRIVATE -lineinfo)


add_executable(hgemm_v2_wmma_m16n16k16_mma4x2_kernel hgemm_v2_wmma_m16n16k16_mma4x2_kernel.cu)
target_link_libraries(hgemm_v2_wmma_m16n16k16_mma4x2_kernel PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(hgemm_v2_wmma_m16n16k16_mma4x2_kernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()


add_executable(hgemm_v3_wmma_m16n16k16_mma4x2_warp2x4_kernel hgemm_v3_wmma_m16n16k16_mma4x2_warp2x4_kernel.cu)
target_link_libraries(hgemm_v3_wmma_m16n16k16_mma4x2_warp2x4_kernel PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(hgemm_v3_wmma_m16n16k16_mma4x2_warp2x4_kernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(hgemm_v4_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async_kernel hgemm_v4_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async_kernel.cu)
target_link_libraries(hgemm_v4_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async_kernel PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(hgemm_v4_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async_kernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()