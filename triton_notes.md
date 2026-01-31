#### triton.jit

> 使用 Triton 编译器的 JIT 编译函数的装饰器。

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask) 
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask) # 相加的结果存入output

grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
```



- 此函数将在 GPU 上编译和运行。它只能访问以下内容：
  - Python 原语，
  - Triton 包内的内置函数，
  - 此函数的参数，
  - 其他 JIT 编译的函数。

#### triton.language.tensor
> 表示一个值或指针的 N 维数组。

```c++
class triton.language.tensor(self, handle, type: dtype)
```

在 Triton 程序中，`tensor` 是最基本的数据结构。`triton.language` 中的大多数函数对 tensors 进行操作并返回。

这里大多数命名的成员函数都是 `triton.language` 中自由函数的重复。例如，`triton.language.sqrt(x)` 等同于 `x.sqrt()`。

`tensor` 还定义了大部分的魔法/双下划线方法，因此可以像写 `x+y`、`x << 2` 等等。



#### triton.language.program_id
> 沿着给定的 axis 返回当前程序实例的 ID
> - axis (int) - 3D 启动网格的轴。必须为 0、1 或 2。

#### triton.language.num_programs
> 返回沿着指定 axis 启动的程序实例的数量。
> -axis (int) - 3D 启动网格的轴。必须为 0、1 或 2。

```c++
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    # starting row of the program
    # 程序起始行
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
```

program_id 即为 grid对应轴上的 block_id

num_programs 即为grid对应轴上的维度范围

```python
# 定义 Grid 为 (X, Y)
grid = (10, 4) 

# 在 Kernel 内部：
pid_x = tl.program_id(0)      # 取值范围 [0, 9]
pid_y = tl.program_id(1)      # 取值范围 [0, 3]

num_p_x = tl.num_programs(0)  # 结果为 10
num_p_y = tl.num_programs(1)  # 结果为 4
```



**BLOCK_SIZE 指的是一个向量/矩阵块的大小，不代表线程数。**

线程数由`num_warps`决定，CUDA 的线程数 = `num_warps * 32`。

| **概念**     | **Triton (逻辑层)**         | **CUDA (物理层)** | **映射关系说明**                                             |
| ------------ | --------------------------- | ----------------- | ------------------------------------------------------------ |
| **任务网格** | `grid` (元组)               | `gridDim`         | **直接映射**。`grid=(1024,)` 在 CUDA 中就是启动 1024 个 Thread Block。 |
| **计算单位** | **Program** (由 `pid` 标识) | **Thread Block**  | 一个 Triton Program 对应一个 CUDA Thread Block。             |
| **数据分块** | `BLOCK_SIZE`                | **无直接对应**    | **核心差异**。Triton 的 `BLOCK_SIZE` 指的是一个向量/矩阵块的大小，不代表线程数。 |
| **线程配置** | `num_warps`                 | `blockDim`        | **物理映射**。CUDA 的线程数 = `num_warps * 32`。             |











#### triton.language.load

> 返回 1 个数据张量，其值从由指针所定义的内存位置处加载：
>
> 

```c++
triton.language.load(pointer, mask=None, other=None, boundary_check=(), padding_option='', cache_modifier='', eviction_policy='', volatile=False)
```




#### triton.language.store



