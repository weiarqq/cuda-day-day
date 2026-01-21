
每个线程正常处理结果矩阵的一个元素，即一行 一列相乘加
nvcc -o bin/sgemm_v0 sgemm_v0_baseline.cu

使用共享内存，把当前block需要计算的行和列加载到共享内存，提升读取效率
nvcc -o bin/sgemm_v1 sgemm_v1_shared_memory.cu

每次读取当前block需要的行列 都需要的等待耗时过长，不利于延迟隐藏，每次只加载一段到共享内存，然后计算，再加载下一部分
nvcc -o bin/sgemm_v2 sgemm_v2_shared_memory_sliding_windows.cu

继续优化延迟隐藏，增加每个线程的计算量，每个线程计算结果矩阵的多个元素，以前计算一行 一列 扩展到 计算 多行多列
nvcc -o bin/sgemm_v3 sgemm_v3_increase_work_of_per_thread.cu

使用 reinterpret_cast<float4 *>(&(pointer))[0] 每次从global memory一次读取四个元素
 nvcc -o bin/sgemm_v4 sgemm_v4_using_float4.cu

前面是拿A的一行 乘以B的一列，能不能先拿A的一个元素，把它该乘的先乘完，这样不用每次都要读取
 nvcc -o bin/sgemm_v5 sgemm_v5_register_outer_product.cu

使用
 nvcc -o bin/sgemm_v6 sgemm_v6_register_outer_product_float4.cu

把B转置，这样B也能按行读
 nvcc -o bin/sgemm_v7 sgemm_v7_A_smem_transpose.cu

双缓存，在第一块计算的时候，同时加载第二块的数据
 nvcc -o bin/sgemm_v8 sgemm_v8_double_buffer.cu
