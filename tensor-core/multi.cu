#include "common.h"
#include <mma.h>
using namespace nvcuda;

// 定义 Tensor Core 操作的维度 (16x16x16 是常用规格)
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void wmma_ker(half* a, half* b, float* c)
{
    // 1. 声明片段 (Fragments)，这些数据将存储在 Tensor Core 寄存器中
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // 2. 初始化累加器片段为 0
    wmma::fill_fragment(acc_frag, 0.0f);

    // 3. 将数据从显存加载到片段中
    // 假设 a, b 是输入矩阵，这里简化了索引逻辑
    wmma::load_matrix_sync(a_frag, a, 16);
    wmma::load_matrix_sync(b_frag, b, 16);

    // 4. 执行矩阵乘加操作 (MMA) - 核心调用！
    // 这一行代码会触发 Tensor Core 硬件加速
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

    // 5. 将结果存回显存
    wmma::store_matrix_sync(c, acc_frag, 16, wmma::mem_row_major);
}

int main()
{
    // 矩阵维度 2^14 = 16384
    const int M = 1 << 14;
    const int N = 1 << 14;
    const int K = 1 << 14;

    printf("矩阵规模: %d x %d x %d\n", M, N, K);

    // 1. 分配主机内存并初始化
    size_t size_a = (size_t)M * K * sizeof(half);
    size_t size_b = (size_t)K * N * sizeof(half);
    size_t size_c = (size_t)M * N * sizeof(float);
    half* h_a = new half[M * N];
    half* h_b = new half[K * N];
    float* h_c = new float[M * N];

    // 初始化 A 和 B 为随机数 (此处简化为 1.0)
    for (int i = 0; i < M * K; ++i)
        h_a[i] = __float2half(1.0f);
    for (int i = 0; i < K * N; ++i)
        h_b[i] = __float2half(1.0f);

    // 2. 分配显存
    half *d_a, *d_b;
    float* d_c;
    CHECK(cudaMalloc(&d_a, size_a));
    CHECK(cudaMalloc(&d_b, size_b));
    CHECK(cudaMalloc(&d_c, size_c));

    // 3. 拷贝数据到显存
    CHECK(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_c, 0, size_c));

    // 4. 配置执行参数
    // 每个 Warp 处理 16x16 的块
    // 每个 Block 配置为 128x4 线程（包含 16 个 Warp），即处理 (4*16)x(4*16) = 64x64 的区域
    dim3 blockDim(32, 16);
    dim3 gridDim((M + (WMMA_M * blockDim.x / 32) - 1) / (WMMA_M * blockDim.x / 32),
        (N + (WMMA_N * blockDim.y) - 1) / (WMMA_N * blockDim.y));

    // 5. 性能计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    wmma_ker<<<gridDim, blockDim>>>(d_a, d_b, d_c);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 6. 计算 TFLOPS
    double tflops = (2.0 * M * N * K) / (milliseconds / 1000.0) / 1e12;
    printf("耗时: %.2f ms\n", milliseconds);
    printf("吞吐量: %.2f TFLOPS\n", tflops);

    // 7. 拷贝回结果并验证 (仅验证第一个元素)
    CHECK(cudaMemcpy(h_c, d_c, sizeof(float), cudaMemcpyDeviceToHost));
    printf("结果验证 C[0]: %.2f (预期: %.2f)\n", h_c[0], (float)K);

    // 释放资源
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}