#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <vector>

#define CHECK(call)                                                             \
    {                                                                           \
        const cudaError_t error = call;                                         \
        if (error != cudaSuccess) {                                             \
            printf("Error: %s:%d", __FILE__, __LINE__);                         \
            printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
            exit(1);                                                            \
        }                                                                       \
    }
void initialData(float* ip, int size)
{
    int i;
    for (i = 0; i < size; i++) {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

void checkResult(float* hostRef, float* gpuRef, const int N)
{
    double epsilon = 1.0e-1;
    int match = 1;
    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n",
                hostRef[i], gpuRef[i], i);
            break;
        }
    }
    if (match)
        printf("Arrays match.\n\n");
    return;
}

double cpuSecond()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void cpu_gemm(float* A, float* B, float* C, const int M, const int N, const int K)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float temp = 0.f;
            for (int k = 0; k < K; k++) {
                temp += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = temp;
        }
    }
}

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&pointer))[0]
template <int M_NUM_PER_BLOCK, int N_NUM_PER_BLOCK, int K_NUM_PER_BLOCK, int NUM_PER_THREAD>
__global__ void sgemm(float* A, float* B, float* C, const int M, const int N, const int K)
{

    float* A_data = A + blockIdx.y * M_NUM_PER_BLOCK * K;
    float* B_data = B + blockIdx.x * N_NUM_PER_BLOCK;

    __shared__ float smem_A[M_NUM_PER_BLOCK][K_NUM_PER_BLOCK];
    __shared__ float smem_B[K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];

    float temp[NUM_PER_THREAD] = { 0.f };
    for (int index = 0; index < K; index += K_NUM_PER_BLOCK) {

        FETCH_FLOAT4(smem_A[threadIdx.y][threadIdx.x * NUM_PER_THREAD]) = FETCH_FLOAT4(A_data[threadIdx.y * K + index + threadIdx.x * NUM_PER_THREAD]);
        FETCH_FLOAT4(smem_B[threadIdx.y][threadIdx.x * NUM_PER_THREAD]) = FETCH_FLOAT4(B_data[(threadIdx.y + index) * N + threadIdx.x * NUM_PER_THREAD]); // 必须按行优先 来 单线程计算NUM_PER_THREAD个; 比如连续内存
        __syncthreads();

        for (int j = 0; j < NUM_PER_THREAD; j++) {
            for (int k = 0; k < K_NUM_PER_BLOCK; k++) {
                temp[j] += smem_A[threadIdx.y][k] * smem_B[k][threadIdx.x * NUM_PER_THREAD + j];
            }
        }
        __syncthreads();
    }
    float* C_ptr = C + blockIdx.y * M_NUM_PER_BLOCK * N + blockIdx.x * N_NUM_PER_BLOCK;
    for (int q = 0; q < NUM_PER_THREAD; q++)
        C_ptr[threadIdx.y * N + threadIdx.x * NUM_PER_THREAD + q] = temp[q];
}

int main(int argc, char** argv)
{

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting reduction at \n", argv[0]);
    printf("device %d: %s \n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // 共享内存放不下这么大数据 改为 128
    int m = 1024;
    int n = 1024;
    const int k = 1024;

    constexpr int M_NUM_PER_BLOCK = 32;
    constexpr int N_NUM_PER_BLOCK = 32;
    constexpr int K_NUM_PER_BLOCK = 32;
    constexpr int NUM_PER_THREAD = 4;

    dim3 block(8, 32); // X维度 一个线程读取4个元素 并计算4个元素，所以是 8 , 8*NUM_PER_THREAD = M_NUM_PER_BLOCK
    dim3 grid(m / N_NUM_PER_BLOCK, n / M_NUM_PER_BLOCK);

    std::vector<float> h_A(m * k);
    std::vector<float> h_B(n * k);
    std::vector<float> h_C(m * n, 0);
    std::vector<float> hd_C(m * n, 0);

    initialData(h_A.data(), m * k);
    initialData(h_B.data(), n * k);

    cpu_gemm(h_A.data(), h_B.data(), h_C.data(), m, n, k);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    CHECK(cudaMalloc((void**)&d_A, m * k * sizeof(float)));
    CHECK(cudaMalloc((void**)&d_B, n * k * sizeof(float)));
    CHECK(cudaMalloc((void**)&d_C, m * n * sizeof(float)));

    CHECK(cudaMemcpy(d_A, h_A.data(), m * k * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B.data(), k * n * sizeof(float), cudaMemcpyHostToDevice));

    double iStart, iElaps;

    for (int q = 0; q < 10; q++) {

        iStart = cpuSecond();
        sgemm<M_NUM_PER_BLOCK, N_NUM_PER_BLOCK, K_NUM_PER_BLOCK, NUM_PER_THREAD><<<grid, block>>>(d_A, d_B, d_C, m, n, k);
        CHECK(cudaDeviceSynchronize());
        iElaps = cpuSecond() - iStart;
        CHECK(cudaMemcpy(hd_C.data(), d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));
        checkResult(h_C.data(), hd_C.data(), m * n);
        printf("num:%d\t%s\t<<<grid (%d %d) block (%d %d)>>>\telapsed %fsec\t\n", k, argv[0], grid.x, grid.y, block.x, block.y, iElaps);
        // std::fill(h_C.begin(), h_C.end(), 0);
        std::fill(hd_C.begin(), hd_C.end(), 0);
        CHECK(cudaMemset(d_C, 0, m * n * sizeof(float)));
    }
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}