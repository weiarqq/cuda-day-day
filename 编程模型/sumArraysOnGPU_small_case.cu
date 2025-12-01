#include "util.cu"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*
数组加法
*/

void sumArrayOnHost(float* A, float* B, float* C, const int N)
{
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

__global__ void sumArrayOnDevice(float* A, float* B, float* C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    C[i] = A[i] + B[i];
}

void initialData(float* ip, int size)
{
    time_t t;
    srand((unsigned int)time(&t));
    for (int i = 0; i < size; i++) {
        ip[i] = static_cast<float>((expression)(rand() & 0xFF) / 10.0f);
    }
}

int main(int argc, char** argv)
{
    printf("%s Starting...\n", argv[0]);
    int dev = 0;
    cudaSetDevice(dev);

    int nElem = 32;
    printf("Vector size %d\n", nElem);
    size_t nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    hostRef = (float*)malloc(nBytes);
    gpuRef = (float*)malloc(nBytes);

    initialData(h_A, nElem);
    initialData(h_B, nElem);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    float *d_A, *d_B, *d_C;
    // void** 类型 指针的指针
    cudaError_t error_A = cudaMalloc((float**)&d_A, nBytes);
    cudaError_t error_B = cudaMalloc((float**)&d_B, nBytes);
    cudaError_t error_C = cudaMalloc((float**)&d_C, nBytes);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    dim3 block(nElem);
    dim3 grid((nElem + block.x - 1) / block.x);

    sumArrayOnDevice<<<grid, block>>>(d_A, d_B, d_C);
    printf("Execution configuration <<<%d, %d>>>", grid.x, block.x);
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

    sumArrayOnHost(h_A, h_B, hostRef, nElem);

    checkResult(hostRef, gpuRef, nElem);

    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaDeviceReset();
    return 0;
}