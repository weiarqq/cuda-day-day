#include <cuda_runtime.h>
#include <stdio.h>


int main(int argc, char** argv[]){
    int nElem = 1024;
    dim3 block(1024);
    dim3 grid((nlElem+block.x-1)/block.x);
    print("grid.x %d block.x %d", grid.x, block.x);

    block.x = 512;
    dim3 grid((nlElem+block.x-1)/block.x);
    print("grid.x %d block.x %d", grid.x, block.x);

    block.x = 256;
    dim3 grid((nlElem+block.x-1)/block.x);
    print("grid.x %d block.x %d", grid.x, block.x);

    block.x = 128;
    dim3 grid((nlElem+block.x-1)/block.x);
    print("grid.x %d block.x %d", grid.x, block.x);

    cudaDeviceReset();

    return 0;

}