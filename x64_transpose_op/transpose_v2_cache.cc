#include "mytranspose.h"

/*
 * 利用cache line（读优先）
*/

void Transpose_v2_cache(const float* input, float* output, size_t M, size_t N)
{
    const size_t K = sizeof(float);                 // 4B
    size_t step = BLOCK / K;                        // 16个
    int m = 0, n = N, limm = M - (M % step), limn = N - (N % step);
    int i = 0, j = 0, idxin = 0, idxout = 0;
    int a = 0, b = 0, tmp = step * N;

    // 按block计算
    for(m = 0, a = 0; m < limm; m += step, a += tmp)
    {
        for(n = 0, b = 0; n < limn; n += step, b += tmp)
        {
            // 单个 step x step block
            int limRow = m + step, limCol = n + step;
            for(i = m, idxin = a; i < limRow; ++i)
            {
                for(j = n, idxout = b; j < limCol; ++j)
                {
                    output[idxout + i] = input[idxin + j];
                    idxout += N;
                }
                idxin += N;
            }
        }
    }

    // 计算剩余 
    b = idxout;
    if(n != N)
    {
        for(i = 0, idxin = 0; i < m; ++i)
        {
            for(size_t j = n, idxout = b; j < N; ++j)
            {
                output[idxout + i] = input[idxin + j];
                idxout += N;
            }
            idxin += N;
        }
    }
    
    for(i = m; i < M; ++i)
    {
        for(j = 0, idxout = 0; j < N; ++j)
        {
            output[idxout + i] = input[idxin + j];
            idxout += N;
        }
        idxin += N;
    }
}