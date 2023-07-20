# DeepLearningOp

Implementation of some deep learning operators. 

## 运行方式

直接执行每个子目录下的脚本文件即可。

## 系统信息

```
OS:         Ubuntu 22.04.1 LTS X64
Linux:      5.19.0-46-generic
Compiler:   g++ 11.3.0

CPU:        Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz
4核8线程，单核 L1d-cache 32K
8路组相联，block_size = 64B，共64组。
指令集支持：avx、avx2、sse、sse2、ssse3、sse4_1、sse4_2、sse4a
``` 

## Layernorm

### 思路

* 算法选择
    - `naive`: two pass 算法，先求均值再求方差，数值稳定性好。
    - default: single pass 算法，直接一次遍历求出均值和方差，数值稳定性在极少数情况不稳定。
    - double:  高精度版的 single pass 算法，避免float类型数的大数吃小数问题，精度很高，数值稳定。
    - **subk**: 根据均值不变性，改良的single pass算法，在保证default算法速度的情况下，保证了数值稳定性。

* 优化点
    - `switch`语句设定mask，减少分支跳转。
    - `maskload`/`maskstore` 处理 corner case。
    - 将单个batch的计算放入一个循环，增强空间局部性。
    - loop unrolling。

## Transpose

### 思路

* 仅针对float类型矩阵
    - v1_base：朴素版本，作为baseline。
    - v2_cache：L1d-cache是按块读入的，块大小为cache line长度，即64B，可容纳16个float。为增加缓存命中率，希望每次读入的block都得到利用，故**分块**读入并转置，块大小即为16x16的float。
    - v3_simd：在分块访问的基础上，256bit的ymm寄存器可容纳8个float，故每个块内继续拆分可并行计算的小块计算。
