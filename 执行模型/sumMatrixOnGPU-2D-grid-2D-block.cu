#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "common.cuh"

/*
用CPU计时器计时
*/

void sumMatrixOnHost(float* A, float* B, float* C, const int nx, const int ny)
{
    float *ia = A;
    float *ib = B;
    float *ic = C;
    for(int iy=0; iy < ny; iy ++){
        for (int ix = 0; ix < nx; ix++) {
            ic[ix] = ia[ix] + ib[iElaps];
        }
        ia += nx;
        ib += nx;
        ic += nx;
    }
}

__global__ void sumMatrixOnGPU2D(float* A, float* B, float* C, const in nx, const int ny)
{
    // blockIdx threadIdx都为 unsigned int 类型
    // unsigned int 不会负数溢出
    // GPU对无符号整数运算对支持更高效
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = ix + iy*nx;
    if(ix < nx && iy <ny)
        C[idx] = A[idx] + B[idx];
}

void initialData(float* ip, int size)
{
    time_t t;
    srand((unsigned int)time(&t));
    for (int i = 0; i < size; i++) {
        ip[i] = static_cast<float>((rand() & 0xFF) / 10.0f);
    }
}

int main(int argc, char** argv)
{
    printf("%s Starting...\n", argv[0]);
    int dev = 0;
    double iStart, iElaps;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Use Device: %d %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int nx = 1 << 14;
    int ny = 1 << 14;
    int nElem = nx * ny;
    printf("Vector size %d\n", nElem);
    size_t nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    hostRef = (float*)malloc(nBytes);
    gpuRef = (float*)malloc(nBytes);

    iStart = cpuSecond();
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    iElaps = cpuSecond() - iStart;

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    iStart = cpuSecond();
    sumMatrixOnHost(h_A, h_B, hostRef, nElem);
    iElaps = cpuSecond() - iStart;
    printf("sumArraysOnHost Time elaspsed %f sec \n", iElaps);

    float *d_A, *d_B, *d_C;
    // void** 类型 指针的指针
    cudaMalloc((void**)&d_A, nBytes);
    cudaMalloc((void**)&d_B, nBytes);
    cudaMalloc((void**)&d_C, nBytes);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1)/block.y);

    iStart = cpuSecond();
    sumMatrixOnGPU2D<<<grid, block>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("sumArraysOnDevice <<<%d, %d>>> Time elaspsed %f sec \n", grid.x, block.x, iElaps);
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

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