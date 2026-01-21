



set(COMMON_FILES common.cpp)
# 构建可执行文件
add_executable(reduce reduce.cu ${COMMON_FILES})
add_cudadevrt(reduce)
target_compile_options(reduce PRIVATE -lineinfo)
