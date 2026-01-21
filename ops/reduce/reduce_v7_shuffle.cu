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
#define WARP_SIZE 32

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
    float sum = 0;
    int tid = threadIdx.x;
    // 将线程块两两分组，组内相加放入共享内存，gourp的数据范围是(blockIdx.x * blockDim.x * 2,  blockIdx.x * blockDim.x * 2 + blockDim.x)
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    if (idx >= n)
        return;

#pragma unroll
    for (int k = 0; k < NUM_PER_THREAD; k++) {
        sum += g_idata[tid + k * BLOCK_SIZE];
    }

    extern __shared__ int smem[];

    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;

    if (BLOCK_SIZE >= 32)
        sum += __shfl_down_sync(0xfffffff, sum, 16);
    if (BLOCK_SIZE >= 16)
        sum += __shfl_down_sync(0xfffffff, sum, 8);
    if (BLOCK_SIZE >= 8)
        sum += __shfl_down_sync(0xfffffff, sum, 4);
    if (BLOCK_SIZE >= 4)
        sum += __shfl_down_sync(0xfffffff, sum, 2);
    if (BLOCK_SIZE >= 2)
        sum += __shfl_down_sync(0xfffffff, sum, 1);
    if (lane_id == 0)
        smem[warp_id] = sum;

    __syncthreads();
    sum = threadIdx.x < BLOCK_SIZE / WARP_SIZE ? smem[lane_id] : 0;
    // WARP_SIZE = 32 BLOCK_SIZE最大值为1024 1024/32=32 所以 warp的数量不会超过32，则最终的结果可以放到一个warp里
    if (warp_id == 0) {
        if (BLOCK_SIZE / WARP_SIZE >= 32)
            sum += __shfl_down_sync(0xfffffff, sum, 16);
        if (BLOCK_SIZE / WARP_SIZE >= 16)
            sum += __shfl_down_sync(0xfffffff, sum, 8);
        if (BLOCK_SIZE / WARP_SIZE >= 8)
            sum += __shfl_down_sync(0xfffffff, sum, 4);
        if (BLOCK_SIZE / WARP_SIZE >= 4)
            sum += __shfl_down_sync(0xfffffff, sum, 2);
        if (BLOCK_SIZE / WARP_SIZE >= 2)
            sum += __shfl_down_sync(0xfffffff, sum, 1);
    }
    if (tid == 0)
        g_odata[blockIdx.x] = sum;
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
    const int block_num = 1024;
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
        reduce<DIM, num_per_thread><<<grid, block>>>(d_idata, d_odata, size);
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