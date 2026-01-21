#include <cassert>
#include <cstdio>
#include <cstring>
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

#define DIM 256

double cpuSecond()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

// BLOCK_SIZE 确定的情况下，我们可以考虑 BLOCK_NUM和 每个线程处理多少数据 进行优化
template <unsigned int BLOCK_SIZE, int NUM_PER_THREAD>
__global__ void reduce(float* g_idata, float* g_odata, unsigned int n)
{
    __shared__ int smem[DIM];
    int tid = threadIdx.x;
    // 将线程块两两分组，组内相加放入共享内存，gourp的数据范围是(blockIdx.x * blockDim.x * 2,  blockIdx.x * blockDim.x * 2 + blockDim.x)
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    if (idx >= n)
        return;
    smem[tid] = 0;

#pragma unroll
    for (int k = 0; k < NUM_PER_THREAD; k++) {
        smem[tid] += g_idata[tid + k * BLOCK_SIZE];
    }

    __syncthreads();
    // BLOCK_SIZE不能超过 1024
    if (BLOCK_SIZE >= 512) {
        if (tid < 256) {
            smem[tid] += smem[tid + 256];
        }
        __syncthreads();
    }

    if (BLOCK_SIZE >= 256) {
        if (tid < 128) {
            smem[tid] += smem[tid + 128];
        }
        __syncthreads();
    }

    if (BLOCK_SIZE >= 128) {
        if (tid < 64) {
            smem[tid] += smem[tid + 64];
        }
        __syncthreads();
    }

    if (tid < 32) {
        // 当确定 线程(tid)来自同一线程束时，不需要__syncthreads(), 每条指令都是同步的
        volatile int* vsmem = smem; // 禁止编译器缓存优化 使用unroll展开时，编译器会进行优化，提前加载 vsmem[tid + 32], vsmem[tid + 16]等，则会使用旧值
        if (BLOCK_SIZE >= 64)
            vsmem[tid] += vsmem[tid + 32];
        if (BLOCK_SIZE >= 32)
            vsmem[tid] += vsmem[tid + 16];
        if (BLOCK_SIZE >= 16)
            vsmem[tid] += vsmem[tid + 8];
        if (BLOCK_SIZE >= 8)
            vsmem[tid] += vsmem[tid + 4];
        if (BLOCK_SIZE >= 4)
            vsmem[tid] += vsmem[tid + 2];
        if (BLOCK_SIZE >= 2)
            vsmem[tid] += vsmem[tid + 1];
    }

    if (tid == 0)
        g_odata[blockIdx.x] = smem[0];
}

// unroll 循环展开

// unroll 完全展开

//

// shuffle

int main(int argc, char** argv)
{

    // set up device
    int dev = 0;
    float gpu_sum = 0.0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting reduction at \n", argv[0]);
    printf("device %d: %s \n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    const int size = 1 << 24; // total number of elements to reduceNeighbored
    const int block_num = 512;
    const int num_per_thread = size / block_num / DIM;
    dim3 block(DIM, 1);
    // block数减半，一个线程 读取两个block的数据
    dim3 grid(block_num, 1);

    std::vector<float> h_idata(size);
    std::vector<float> h_odata(grid.x);
    float cpu_sum = 0;

    size_t nbytes = size * sizeof(float);

    // allocate device memory
    float* d_idata = NULL;
    float* d_odata = NULL;
    CHECK(cudaMalloc((void**)&d_idata, size * sizeof(float)));
    CHECK(cudaMalloc((void**)&d_odata, grid.x * sizeof(float)));

    for (int k = 0; k < 10; k++) {
        for (int i = 0; i < size; i++) {
            h_idata[i] = 1;
            cpu_sum += 1;
        }

        double iStart, iElaps;
        CHECK(cudaMemcpy(d_idata, h_idata.data(), nbytes, cudaMemcpyHostToDevice));
        iStart = cpuSecond();
        reduce<DIM, num_per_thread><<<grid, block, (block.x / 32) * sizeof(int)>>>(d_idata, d_odata, size);
        CHECK(cudaDeviceSynchronize());
        iElaps = cpuSecond() - iStart;
        CHECK(cudaMemcpy(h_odata.data(), d_odata, grid.x * sizeof(float), cudaMemcpyDeviceToHost));
        gpu_sum = 0;
        for (int i = 0; i < grid.x; i++)
            gpu_sum += h_odata[i];

        assert(cpu_sum == gpu_sum && "cpu_sum != gpu_sum");

        printf("num:%d\t%s\t<<<grid %d block %d>>>\telapsed %fsec\t\n", k, argv[0], grid.x, block.x, iElaps);
        std::fill(h_idata.begin(), h_idata.end(), 0);
        std::fill(h_odata.begin(), h_odata.end(), 0);
        cudaMemset(d_odata, 0, nbytes);
    }
    CHECK(cudaFree(d_idata));
    CHECK(cudaFree(d_odata));

    CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}