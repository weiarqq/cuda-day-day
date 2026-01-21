all:
	nvcc -o bin/sgemm_v0 sgemm_v0_baseline.cu
	nvcc -o bin/sgemm_v1 sgemm_v1_shared_memory.cu
# nvcc -o bin/sgemm_v2 sgemm_v2_shared_memory_sliding_windows.cu
# nvcc -o bin/sgemm_v3 sgemm_v3_increase_work_of_per_thread.cu
# nvcc -o bin/sgemm_v4 sgemm_v4_using_float4.cu
# nvcc -o bin/sgemm_v5 sgemm_v5_register_outer_product.cu
# nvcc -o bin/sgemm_v6 sgemm_v6_register_outer_product_float4.cu
# nvcc -o bin/sgemm_v7 sgemm_v7_A_smem_transpose.cu
# nvcc -o bin/sgemm_v8 sgemm_v8_double_buffer.cu