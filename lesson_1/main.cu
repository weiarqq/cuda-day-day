#include <stdio.h>

__global__ void print_hw(void)
{
    printf("GPU: hello world!");
}

int main()
{
    printf("CPU: hello world");
    print_hw<<<1, 10>>>();
    cudaDeviceReset();
}