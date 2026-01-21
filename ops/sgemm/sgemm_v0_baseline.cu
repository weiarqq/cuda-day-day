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

__global__ void sgemm(float* A, float* B, float* C, const int M, const int N, const int K)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    float* A_data = A + idy * K;
    float* B_data = B + idx;

    if (idx >= N || idy >= M)
        return;

    float temp = 0.f;
    for (int k = 0; k < K; k++) {
        temp += A_data[k] * B_data[k * N];
    }

    C[idy * N + idx] = temp;
}

#define P 32

int main(int argc, char** argv)
{

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting reduction at \n", argv[0]);
    printf("device %d: %s \n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int m = 2048;
    int n = 2048;
    int k = 2048;

    int p = 128;
    dim3 block(m / p, n / p);
    dim3 grid((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);

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
        sgemm<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
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