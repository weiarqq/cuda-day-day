### CUDA 入门



**hello world**

```c++
//main.cu
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
```

编译

```shell
$ nvcc main.cu -o main
```





`__global__`:是告诉编译器这个是个可以在设备上执行的核函数

```c++
hello_world<<<1,10>>>();
```

这句话C语言中没有’<<<>>>’是对设备进行配置的参数，也是CUDA扩展出来的部分。

```c++
cudaDeviceReset();
```

这句话如果没有，则不能正常的运行，因为这句话包含了隐式同步，GPU和CPU执行程序是异步的，核函数调用后成立刻会到主机线程继续，而不管GPU端核函数是否执行完毕，所以上面的程序就是GPU刚开始执行，CPU已经退出程序了，所以我们要等GPU执行完了，再退出主机线程。

一般CUDA程序分成下面这些步骤：

1. 分配GPU内存
2. 拷贝内存到设备
3. 调用CUDA内核函数来执行计算
4. 把计算完成数据拷贝回主机端
5. 内存销毁