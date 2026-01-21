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

// 相邻元素相加
// 将数据拷贝到共享内存，由于涉及到数据的多次读取和写入，中间过程放入共享内存 提升读写效率
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

    for (auto stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0)
            smem[tid] += smem[tid + stride];

        __syncthreads();
    }
    if (tid == 0)
        g_odata[blockIdx.x] = smem[0];
}

// 无线程束分化 按线程id序 0 1 2 3 4 来读取相隔位置的数据

// 无 bank conflict

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

        assert(cpu_sum == gpu_sum && "cpu_sum != gpu_sum");

        printf("num:%d\t%s\t<<<grid %d block %d>>>\telapsed %fsec\t\n", k, argv[0], grid.x, block.x, iElaps);
        std::fill(h_idata.begin(), h_idata.end(), 0);
        std::fill(h_odata.begin(), h_odata.end(), 0);
        cudaMemset(d_odata, 0, nbytes);
        cpu_sum = 0;
    }
    CHECK(cudaFree(d_idata));
    CHECK(cudaFree(d_odata));

    CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}