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

template <int BLOCK_SIZE, int BLOCK_STRIDE>
__global__ void sgemm(float* A, float* B, float* C, const int M, const int N, const int K)
{

    float* A_data = A + blockIdx.y * blockDim.y * BLOCK_STRIDE * K;
    float* B_data = B + blockIdx.x * blockDim.x * BLOCK_STRIDE;

    constexpr unsigned int stride = BLOCK_SIZE * BLOCK_STRIDE;

    __shared__ float smem_A[stride][stride];
    __shared__ float smem_B[stride][stride];
    // 此处 一个block需要处理 BLOCK_SIZE * BLOCK_SIZE的元素，需要加载 A：BLOCK_SIZE*K 和 B: K*BLOCK_SIZE
    float temp[BLOCK_STRIDE][BLOCK_SIZE] = { 0.f };
    // 可用资源 一个block BLOCK_SIZE * BLOCK_SIZE 的线程数，用来 加载 BLOCK_SIZE*K + K*BLOCK_SIZE的元素，则每个线程加载 A的K/BLOCK_SIZE + B的 K/BLOCK_SIZE的元素
    for (int index = 0; index < K; index += stride) {
        for (int i = 0; i < BLOCK_STRIDE; i++) {
            for (int j = 0; j < BLOCK_STRIDE; j++) {
                smem_A[threadIdx.y + i * BLOCK_SIZE][threadIdx.x + j * BLOCK_SIZE] = A_data[(threadIdx.y + i * BLOCK_SIZE) * K + (threadIdx.x + j * BLOCK_SIZE) + index];
                smem_B[threadIdx.y + i * BLOCK_SIZE][threadIdx.x + j * BLOCK_SIZE] = B_data[(threadIdx.y + i * BLOCK_SIZE + index) * N + threadIdx.x + j * BLOCK_SIZE];
            }
        }

        __syncthreads();
        for (int i = 0; i < BLOCK_STRIDE; i++) {
            for (int j = 0; j < BLOCK_STRIDE; j++) {
                for (int k = 0; k < stride; k++) {
                    temp[i][j] += smem_A[threadIdx.y + i * BLOCK_SIZE][k] * smem_B[k][threadIdx.x + j * BLOCK_SIZE];
                }
            }
        }
        __syncthreads();
    }
    // 每个block处理 BLOCK_STRIDE个block的数据
    // C + (blockIdx.y * blockDim.y) * BLOCK_STRIDE * N + (blockDim.x * stride)
    float* C_ptr = C + (blockIdx.y * stride) * N + (blockIdx.x * stride);
    for (int i = 0; i < BLOCK_STRIDE; i++) {
        for (int j = 0; j < BLOCK_STRIDE; j++) {
            C_ptr[(threadIdx.y + i * BLOCK_SIZE) * N + threadIdx.x + j * BLOCK_SIZE] = temp[i][j];
        }
    }
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
    const int m = 1024;
    const int n = 1024;
    const int k = 1024;
    const int STRIDE = 2;
    const int BLOCK_SIZE = 16;

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((m + block.x - 1) / block.x / STRIDE, (n + block.y - 1) / block.y / STRIDE);

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
        sgemm<BLOCK_SIZE, STRIDE><<<grid, block>>>(d_A, d_B, d_C, m, n, k);
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