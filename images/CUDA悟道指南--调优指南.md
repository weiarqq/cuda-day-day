



## CUDA悟道指南--调优指南



### 一、性能指标

#### Operational Intensity(计算强度)

> 计算密度\计算访存比
> 参考：https://zhuanlan.zhihu.com/p/34204282

**1. 计算平台的两个指标：算力 $\pi$与带宽 $\beta$**

- **算力** $\pi$：也称为计算平台的**性能上限**，指的是一个计算平台倾尽全力每秒钟所能完成的浮点运算数。单位是 `FLOPS` or `FLOP/s`。
								${\color{red}{\pi}} : \textrm{Maximum FLOPs Per Second} $



- **带宽**$\beta$ ：也即计算平台的**带宽上限**，指的是一个计算平台倾尽全力每秒所能完成的内存交换量。单位是`Byte/s`。
     	${\color{red}{\beta}} : \textrm{Maximum Memory Access Per Second} $



- **计算强度上限**$I_{max}$ ：两个指标相除即可得到计算平台的**计算强度上限**。它描述的是在这个计算平台上，单位内存交换最多用来进行多少次计算。单位是`FLOPs/Byte`。
     	$I_{max} = \frac{\pi}{\beta} $

  

> 注：这里所说的“内存”是广义上的内存。对于CPU计算平台而言指的就是真正的内存；而对于GPU计算平台指的则是显存。

---

 **2. 模型的两个指标：计算量 与 访存量**

- **计算量：**指的是输入单个样本（对于CNN而言就是一张图像），模型进行一次完整的前向传播所发生的浮点运算个数，也即模型的**时间复杂度**。单位是 `FLOP` or `FLOPs`。矩阵乘法的计算量公式如下：
					$ \textrm{GEMM Time Complexity} : 2 \cdot K \cdot M \cdot N \quad (\textbf{FLOPS}) $



- **访存量：**指的是输入单个样本，模型完成一次前向传播过程中所发生的内存交换总量，也即模型的**空间复杂度**。在理想情况下（即不考虑片上缓存），模型的访存量就是模型各层权重参数的内存占用（Kernel Mem）与每层所输出的特征图的内存占用（Output Mem）之和。单位是`Byte`。由于数据类型通常为`float32` ，因此需要乘以四。
					$\textrm{GEMM Space Complexity} : M \cdot K  + N \cdot K + M \cdot N \quad (\textbf{Bytes})$



- **模型的计算强度 I ：**由计算量除以访存量就可以得到模型的计算强度，它表示此模型在计算过程中，每`Byte`内存交换到底用于进行多少次浮点运算。单位是`FLOPs/Byte`。可以看到，模计算强度越大，其内存使用效率越高。



- **模型的理论性能 P ：**我们最关心的指标，即模型**在计算平台上所能达到的每秒浮点运算次数（理论值）**。单位是 `FLOPS` or `FLOP/s`。下面我们即将介绍的 Roof-line Model 给出的就是计算这个指标的方法。终于可以进入正题了。

----

**3. Roof-line Model**

其实 Roof-line Model 说的是很简单的一件事：**模型在一个计算平台的限制下，到底能达到多快的浮点计算速度**。更具体的来说，Roof-line Model 解决的，是“**计算量为A且访存量为B的模型在算力为C且带宽为D的计算平台所能达到的理论性能上限E是多少**”这个问题。

---

 **3.1 Roof-line 的形态**

所谓“Roof-line”，指的就是由计算平台的算力和带宽上限这两个参数所决定的“屋顶”形态，如下图所示。

- **算力**决定“屋顶”的高度（绿色线段）
- **带宽**决定“房檐”的斜率（红色线段）

![img](/Users/wangqi/workspace/workspace/cuda-day-day/images/v2-cafb93b9a31fca2d7c84951555762e59_r.png)

---

**3.2 Roof-line 划分出的两个瓶颈区域**



​				$ P = \begin{cases} \beta \cdot I, & ~ when ~~ I < I_{max} \quad {\color{red}{\textbf{Memory Bound}}}\\[2ex] \pi, & ~ when ~~ I \geqslant I_{max} \quad {\color{green}{\textbf{Compute Bound}}} \end{cases} $



**计算瓶颈区域 `Compute-Bound`**

不管模型的计算强度 I 有多大，它的理论性能 P 最大只能等于计算平台的算力 \pi 。当模型的计算强度 I 大于计算平台的计算强度上限 I_{max} 时，模型在当前计算平台处于 `Compute-Bound`状态，即模型的理论性能 P 受到计算平台算力 \pi 的限制，无法与计算强度 I 成正比。但这其实并不是一件坏事，因为从充分利用计算平台算力的角度上看，此时模型已经 100\% 的利用了计算平台的全部算力。可见，计算平台的算力 \pi 越高，模型进入计算瓶颈区域后的理论性能 P 也就越大。



**带宽瓶颈区域 `Memory-Bound`**

当模型的计算强度 I 小于计算平台的计算强度上限 I_{max} 时，由于此时模型位于“房檐”区间，因此模型理论性能 P 的大小完全由计算平台的带宽上限 \beta （房檐的斜率）以及模型自身的计算强度 I 所决定，因此这时候就称模型处于 `Memory-Bound` 状态。可见，在模型处于带宽瓶颈区间的前提下，计算平台的带宽 \beta 越大（房檐越陡），或者模型的计算强度 I 越大，模型的理论性能 P 可呈线性增长。




### 二、性能调优

1. 循环展开



### 五、 性能调优

计算强度和计算密度 计算访问存比

在 CUDA 性能调优中，我们追求的核心目标通常是**最大化吞吐量**和**最小化延迟**。为了达成这个目标，我们需要从 GPU 的硬件特性出发，重点关注以下几个维度的指标。

我们可以将这些指标分为：**指令吞吐、存储带宽、资源利用率**三个大类。

#### 存储指标 (Memory Metrics)

GPU 往往是“存取受限”的（Memory-bound），因此存储指标是最优先关注的。

- **Global Memory Load/Store Efficiency (全局内存存取效率)**：
  - **定义**：实际请求的字节数与总传输字节数的比率。
  - **调优目标**：追求 100%。如果效率低，通常是因为没有实现**合并访问 (Coalesced Access)**，导致硬件浪费了带宽去读取不需要的数据。
- **Memory Bandwidth Utilization (带宽利用率)**：
  - 检查程序是否达到了硬件理论带宽的极限（如 H100 的 3TB/s）。
- **Shared Memory Bank Conflicts (共享内存 Bank 冲突)**：
  - **定义**：当同一个 Warp 中的多个线程访问同一个 Bank 的不同地址时，会发生冲突，导致访问串行化。
  - **调优目标**：尽量消除冲突，保证共享内存的高速并行访问。

#### 计算与指令指标 (Compute & Instruction Metrics)

当存储不再是瓶颈时，计算效率就变得至关重要。

- **Occupancy (活跃线程束占有率)**：
  - **定义**：每个多处理器 (SM) 上活跃的 Warp 数与最大支持 Warp 数的比率。
  - **注意**：虽然高 Occupancy 有助于掩盖延迟，但它不是越高越好。过高的 Occupancy 可能会限制每个线程可用的寄存器数量，反而降低性能。
- **Warp Execution Efficiency (Warp 执行效率)**：
  - **定义**：由于条件分支（if-else）导致的**线程分歧 (Divergence)**。
  - **调优目标**：减少 `if` 分支，确保 Warp 内的 32 个线程尽可能走相同的路径。
- **Floating Point Operations (FLOPS) 利用率**：
  - 观察单精度 (FP32)、半精度 (FP16/BF16) 或 Tensor Core 的利用率。对于深度学习负载，**Tensor Core 利用率**是核心指标。

#### 延迟掩盖 (Latency Hiding)

GPU 依靠海量的线程并行来“掩盖”昂贵的指令或内存延迟。

- **DRAM Latency (显存延迟)**：
  - 如果 Occupancy 足够但性能仍差，可能是因为指令发射后一直在等待内存回传数据。
- **Stall Reasons (阻塞原因)**：
  - 使用 NVIDIA Nsight Compute 分析时，重点看 Warp 阻塞的原因。常见原因包括：
    - **Instruction Fetch**：取指令慢。
    - **Execution Dependency**：当前指令依赖前一指令的结果。
    - **Data Request**：等待存储器操作完成。

#### 资源平衡 (Resource Limits)

调优本质上是在有限的资源中做平衡。

| **资源**               | **对性能的影响**                                            |
| ---------------------- | ----------------------------------------------------------- |
| **Registers (寄存器)** | 每个线程用太多寄存器会降低 Occupancy。                      |
| **Shared Memory**      | 每个 Block 申请过多共享内存也会降低 Occupancy。             |
| **Grid/Block Size**    | 设置不当会导致 SM 负载不均衡或产生尾部效应（Tail effect）。 |

#### 建议的调优路径

1. **使用 Nsight Systems**：先看全局，确认瓶颈是在 CPU、数据传输（PCIe）还是内核执行上。
2. **使用 Nsight Compute**：针对具体的 Kernel 进行精细化分析。
3. **关注 Roofline Model**：通过 Roofline 图判断你的程序是 **Memory-bound** (带宽受限) 还是 **Compute-bound** (计算受限)，这决定了你的优化方向。

------

**您是否已经有特定的代码遇到了性能瓶颈？** 如果有，您可以提供 Kernel 的逻辑或目前的 Nsight 分析结果，我可以帮您分析具体的调优策略。




#### 1. Compute (SM) Throughput（计算吞吐量）

这个指标代表 **Streaming Multiprocessor (SM)** 的综合利用率。

- **定义：** 它不是指某个单一的算术单元，而是 SM 内部所有执行流水线（Pipelines）中**最繁忙的那一个**的利用率百分比。
- **包含内容：**
  - **算术指令：** FP32 (FMA/ALU), FP64, INT, FP16。
  - **特殊单元：** Tensor Cores（张量核心）、SFU（特殊函数单元，如 sin/cos）。
  - **流水线：** 如果你的代码大量执行整数运算，那么 `ALU Throughput` 可能会决定 `Compute Throughput` 的数值。
- **调优意义：** 如果该数值很高（如 >80%），说明你的 Kernel 是**计算受限型 (Compute-Bound)**。此时增加内存带宽不会提升性能，你需要减少计算量或使用更高效的指令（如 Tensor Cores）。

------

#### 2. Memory Throughput（内存吞吐量）

这个指标代表 GPU **存储体系**的综合利用率。

- **定义：** 它是整个内存子系统（从 L1 缓存到 DRAM 显存）中**最繁忙的路径或单元**的利用率百分比。
- **包含内容：**
  - **DRAM (VRAM)：** 访问显存的带宽。
  - **L2 Cache：** L2 缓存的吞吐量。
  - **L1/TEX Cache：** L1 缓存和纹理缓存。
- **特殊逻辑：** 即使你没有访问显存（DRAM Throughput 很低），但如果你的代码由于逻辑原因（如 Bank Conflict）导致 L1 缓存非常忙碌，`Memory Throughput` 的数值依然会很高。
- **调优意义：** 如果该数值很高，说明 Kernel 是**访存受限型 (Memory-Bound)**。你需要优化访存模式（合并访问）、提高缓存命中率或减少数据搬运。



**为什么它们经常一起出现？（Roofline 模型）**

| **场景**       | **表现**              | **瓶颈点**                                                |
| -------------- | --------------------- | --------------------------------------------------------- |
| **计算密集型** | Compute % >> Memory % | 算术运算太多，核心太忙。                                  |
| **访存密集型** | Memory % >> Compute % | 数据传得慢，核心在等数据。                                |
| **延迟受限型** | 两者都很低            | 可能是因为线程数太少，无法掩盖指令延迟（Latency Bound）。 |

![](/Users/wangqi/workspace/workspace/cuda-day-day/images/1768669879204.jpg)



#### 3. Occupancy（占用率）

> ncu --section Occupancy ./app

**占用率 = (实际驻留在 SM 上的活跃线程束数量) / (SM 支持的最大活跃线程束数量)**

简单来说，就是 SM 的“客满程度”。

- **高占用率的作用**：当一个 Warp 因为读内存而阻塞时，SM 的硬件调度器可以瞬间切换到另一个已经准备就绪的 Warp 进场计算。只要 SM 里“待命”的 Warp 足够多，计算单元就能一直保持忙碌，从而“掩盖”掉漫长的访存延迟。

占用率（Occupancy）是 CUDA 性能优化中最常被提及的概念之一。要理解它，我们得先从 GPU 为什么要“多线程并行”说起。

1. 核心矛盾：计算太快，访存太慢，GPU 的运算单元（ALU）速度极快，但访问显存（Global Memory）却非常慢。

- 一次计算可能只需几个时钟周期。
- 一次显存读取可能需要 **400 到 800 个时钟周期**。

如果一个 SM（流式多处理器）里只运行一个 Warp（32个线程），当这个 Warp 发出读取显存的指令后，它就会陷入漫长的等待。在这几百个周期里，SM 的计算单元就只能“闲着”。
 **这就引出了“隐藏延迟（Latency Hiding）”的概念。**

----

Q：占用率越高越好吗？

> 不一定。
>
> 当占用率达到一定程度（通常是 50%~70%）后，延迟通常就已经能被很好地隐藏了。盲目追求 100% 的占用率有时反而有害：
>
> - 如果为了凑占用率而极度压缩每个线程使用的寄存器数量，会导致程序频繁读写慢速内存（Register Spilling），反而让整体速度变慢。

![](/Users/wangqi/workspace/workspace/cuda-day-day/images/1768671048062.jpg)



| **指标名称 (Metric Name)**          | **单位** | **含义解析**                                                 |
| ----------------------------------- | -------- | ------------------------------------------------------------ |
| **Block Limit SM**                  | block    | **SM 硬件限制：** 每个 SM 最多能容纳的线程块数量（纯硬件架构限制，不考虑资源分配）。 |
| **Block Limit Registers**           | block    | **寄存器限制：** 根据你代码中每个线程使用的寄存器数量，计算出每个 SM 最多能跑多少个块。**这个值越小，说明寄存器压力越大。** |
| **Block Limit Shared Mem**          | block    | **共享内存限制：** 根据你分配的 `shared memory` 大小，计算出 SM 能容纳的块上限。 |
| **Block Limit Warps**               | block    | **Warp 总数限制：** SM 有最大活跃 Warp 数限制（如 48 或 64）。根据你每个 Block 里的线程数，算出的块上限。 |
| **Theoretical Active Warps per SM** | warp     | **理论最大活跃 Warp 数：** 在理想状态下，当前 SM 能够同时维持运行的最大 Warp 数量。 |
| **Theoretical Occupancy**           | %        | **理论占用率：** $Theoretical\ Active\ Warps \div Max\ Warps\ per\ SM$。你设置的参数（Grid/Block size）理论上能达到的最高效率。 |
| **Achieved Occupancy**              | %        | **实际占用率：** 程序运行时，SM 实际上平均维持的占用率。这是性能调优的关键指标。 |
| **Achieved Active Warps Per SM**    | warp     | **实际活跃 Warp 数：** 运行时每个 SM 平均并行的 Warp 数量。你的数据是 30.61，说明没跑满。 |

**以上Block Limit 限制的是block的数量，比如 block_size 256，Block Limit Warps=6，因为SM最大warp数量为48，则Block Limit Warps = 48/(256/32)**





#### 4. MemoryWorkloadAnalysis(内存分析)

> ncu --section MemoryWorkloadAnalysis ./bin/app

![](/Users/wangqi/workspace/workspace/cuda-day-day/images/1768711344275.jpg)

| **指标名称 (Metric Name)** | **单位** | **指标含义与分析**                                           |
| -------------------------- | -------- | ------------------------------------------------------------ |
| **Memory Throughput**      | Gbyte/s  | **显存吞吐量**：内核执行期间，显存（DRAM）实际的数据传输速度。17.09 GB/s 对于高性能 GPU 来说不算高。 |
| **Mem Busy**               | %        | **显存忙碌状态**：显存控制器在采样周期内处于工作状态的时间百分比。44.86% 说明显存带宽还有剩余空间。 |
| **Max Bandwidth**          | %        | **最大带宽利用率**：衡量当前吞吐量占理论峰值带宽的比例。     |
| **L1/TEX Hit Rate**        | %        | **L1/纹理缓存命中率**：74.99% 的命中率属于**中等偏上**。说明大部分数据请求在最靠近核心的 L1 缓存中就得到了满足。 |
| **L2 Hit Rate**            | %        | **L2 缓存命中率**：**99.23% 是一个非常高的数值**。这意味着几乎所有未命中 L1 的数据请求都能在 L2 缓存中找到，只有极少数请求需要去访问缓慢的显存。 |
| **Mem Pipes Busy**         | %        | **内存管道忙碌程度**：负责处理内存指令的硬件流水线的占用率。47.57% 说明硬件单元并未遇到严重的阻塞。 |
| **L2 Compression...**      | -        | **L2 压缩相关指标**：这三项（Success Rate, Ratio, Input Sectors）均为 0，说明当前数据没有经过 L2 压缩，或者该硬件/驱动当前未启用压缩功能。 |







#### 5. PM Samping

在 NVIDIA Nsight Compute (NCU) 中，**PM Sampling (Performance Monitor Sampling)** 是一项高级性能分析功能。

简单来说，传统的 NCU 指标（如 `sm__throughput`）通常是针对整个 Kernel 执行过程的**平均值**，而 PM Sampling 允许你以**时间轴（Timeline）**的形式观察指标在 Kernel 运行期间的变化过程。

以下是关于 PM Sampling 的详细解析：

---

1. 核心概念：为什么需要它？

普通的 NCU 报告会告诉你：“这个 Kernel 的 GPU 利用率是 80%”。但这掩盖了过程中的波动：

* **长尾效应 (Tail Effects)：** Kernel 结束前，是否因为部分线程块（Thread Blocks）运行太慢导致大部分 SM 在空转？
* **阶段性行为 (Phases)：** 算法内部是否有明显的阶段（例如：先从内存读数据，再进行密集计算，最后写回）？
* **指标相关性：** 当 DRAM 吞吐量飙升时，计算单元（Compute Pipe）是否因为等待数据而停顿？

**PM Sampling 将这些“平均数”展开成了“动态波动图”。**

---

2. 主要功能与视图

在 NCU GUI 中开启 PM Sampling 后，你会看到一个专门的 **PM Sampling 区域**，主要包含：

A. 性能指标时间轴 (Timeline View)

* **数据采集：** 它会周期性地（按时钟周期或纳秒）对硬件性能计数器进行采样。
* **可视化：** 你可以同时对比多个指标（如 `Compute Throughput` vs `Memory Throughput`）。
* **对齐：** 如果开启了 **Context Switch Trace**，采集的数据会更精准地对齐到当前的 CUDA 上下文，过滤掉系统其他任务的干扰。

B. Warp 状态采样 (Warp States)

这是 CUDA 12.4 / NCU 2024.1 引入的重要增强功能。

* 它可以展示在 Kernel 运行的不同时间点，Warp 处于什么状态（如 `Stall Wait`、`Stall Math Busy`、`Active` 等）。
* **PmSampling_WarpStates** Section 能帮助你发现：Kernel 是在刚开始时卡在内存延迟上，还是在后期因为指令发射受限而变慢。

---
3. 关键参数设置 (CLI)

如果你使用命令行工具 `ncu`，可以通过以下参数控制采样：

| 参数                        | 说明                                                         |
| --------------------------- | ------------------------------------------------------------ |
| `--pm-sampling-interval`    | **采样间隔**。较小值提供更高分辨率，但会增加开销。Turing 架构通常以周期为单位（如 400,000 cycles），Ampere 及之后通常以纳秒为单位（如 1000ns）。 |
| `--pm-sampling-buffer-size` | **缓冲区大小**。如果 Kernel 运行时间极长，需要增大此值以防止数据溢出导致采样中断。 |
| `--pm-sampling-max-passes`  | 采样可能需要多次 Replay 才能收集完所有请求的指标，此参数限制最大 Pass 数。 |
| `--section PmSampling`      | 显式指定收集 PM 采样相关的 Section。                         |

**示例命令：**

```bash
ncu --section PmSampling --pm-sampling-interval 100000 --pm-sampling-buffer-size 50000000 -o my_profile ./my_app

```

---

4. 注意事项与限制

1. **硬件支持：** PM Sampling 对架构有要求，通常需要 NVIDIA **Turing (RTX 20系列)** 或更新的显卡。在一些特定的云计算环境下（如 vGPU），此功能可能被禁用。
2. **性能开销：** 采样会带来额外的运行时开销，并可能导致 Kernel 被多次回放（Replay）。
3. **MIG 支持：** 在 Multi-Instance GPU (MIG) 模式下支持采样，但通常不支持 Context Switch Trace，这可能导致数据在多 Pass 之间对齐时出现细微偏差。
4. **精度：** 对于运行时间极短（如微秒级）的 Kernel，PM Sampling 的意义不大，因为采样间隔可能比 Kernel 本身还长。

---

5. 总结：什么时候用？
* 如果你发现 Kernel 运行比预期慢，但 **Summary** 页面的指标看起来都还行，那就开启 PM Sampling。
* 检查是否存在 **“头轻脚重”**（开始很快，结束很慢）的现象。
* 分析计算和访存是如何在**时间维度**上交叠（Overlap）的。

![](/Users/wangqi/workspace/workspace/cuda-day-day/images/1768712246111.jpg)
















如何更好的利用缓存



如何更好的利用带宽



如何更好的利用内存，小&块->大&慢





提示性能

减少重复

减轻依赖