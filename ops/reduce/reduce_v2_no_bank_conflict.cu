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

// 无 bank conflict
// 分析一下为什么会有 bank conflict, smem长度为 256，bank0的数据有 0, 32, 64, 96, 128, 160, 192, 224
// 由于线程束分化的原因，我们v1改为同一线程束连续线程进行计算, index = tid * 2 * stride， 计算 index和index + stride的和
// 在第一轮中，stride=1 tid = 0 读取数据 0, 1， tid=16 读取数据 32, 33， 则 出现 bank conflict 0, 16线程 读取bank0的0和32,属于不同地址
// 在第二轮中，stride=2 tid = 0 读取数据 0, 2， tid=16 读取数据 64, 66， 则 出现 bank conflict 0, 16线程 读取bank0的0和64,属于不同地址 。。。
// 如何解决呢，我们可以按照倒序，即 stride = stride/2的方式来
// 我们可以分为两种情况，
// 由于 stride为2的幂，则大于等于32时，stride必为32的倍数，则 同一线程读取同一bank的数据
// 当stride小于32时，即 16, 8, 4, 2, 1, 由于 tid < stride（由于数据量是 2 * stride， 一个线程处理两个数据，所以 tid < stride） 线程读取的都是bank的第一轮数据,即(0~31)
// 其实简单点说，stride 小于32时，只有warp0在执行，且读取数据的范围小于32，所以不会出现 bank conflict的情况， 这一部分其实也有优化空间!!!!
// 我们可以控制让同一线程读取同一bank的数据，既然  0, 16线程 读取bank0的（0，32）,（1， 33）冲突,那我们让线程0读取 0, 32，让线程16读取(16, )就可以了
// 即同一线程处理同一bank，这也是为什么线程束和bank都是按照 32来划分的原因
// 实现
//
__global__ void reduce(float* g_idata, float* g_odata, unsigned int n)
{
    __shared__ int smem[DIM];
    int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    float* idata = g_idata + blockIdx.x * blockDim.x;
    smem[tid] = idata[tid];

    __syncthreads();

    for (auto stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0)
        g_odata[blockIdx.x] = smem[0];
}

// 减少使用的线程，减少block OR 减少thread

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

    int size = 1 << 24; // total number of elements to reduceNeighbored

    dim3 block(256, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);

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
        reduce<<<grid, block>>>(d_idata, d_odata, size);
        CHECK(cudaDeviceSynchronize());
        iElaps = cpuSecond() - iStart;
        CHECK(cudaMemcpy(h_odata.data(), d_odata, grid.x * sizeof(float), cudaMemcpyDeviceToHost));
        gpu_sum = 0;
        for (int i = 0; i < grid.x; i++)
            gpu_sum += h_odata[i];
        printf("%f --- %f \n", cpu_sum, gpu_sum);
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