#include <iomanip>
#include "mytranspose.h"

/*
 * 利用cache line（读优先）
*/

// void PrintArr(int size, float* x)
// {
//     int t = size * size;
//     for(int i = 0; i < t; ++i)
//     {
//         if(i % size == 0)
//             cout<< endl;
//         cout << std::setw(5) << x[i] << " ";
//     }
//     cout<< endl;
// }

void Transpose_v3_simd(const float* input, float* output, size_t M, size_t N)
{
    const size_t K = sizeof(float);                 // 4B
    size_t step = BLOCK / K;                        // 16个
    int m = 0, n = N, limm = M - (M % step), limn = N - (N % step);
    int i = 0, j = 0, idxin = 0, idxout = 0;

    // 按block计算
    for(m = 0; m < limm; m += step)
    {
        for(n = 0; n < limn; n += step)
        {
            // 单个 16 x 16 step block
            size_t cnt = YMMLEN / sizeof(float);                    // ymm一次处理8个float
            int limRow = m + step, limCol = n + step;
            for(i = m; i < limRow; i += cnt)
            {
                for(j = n; j < limCol; j += cnt)            
                {
                    float *sptr = (float*)input + N * i + j;        // TODO 乘->加
                    __m256 row0 = _mm256_loadu_ps(sptr);
                    __m256 row1 = _mm256_loadu_ps(++sptr);
                    __m256 row2 = _mm256_loadu_ps(++sptr);
                    __m256 row3 = _mm256_loadu_ps(++sptr);
                    __m256 row4 = _mm256_loadu_ps(++sptr);
                    __m256 row5 = _mm256_loadu_ps(++sptr);
                    __m256 row6 = _mm256_loadu_ps(++sptr);
                    __m256 row7 = _mm256_loadu_ps(++sptr);

                    __m256 t0 = _mm256_unpacklo_ps(row0, row1);
                    __m256 t1 = _mm256_unpackhi_ps(row0, row1);
                    __m256 t2 = _mm256_unpacklo_ps(row2, row3);
                    __m256 t3 = _mm256_unpackhi_ps(row2, row3);
                    __m256 t4 = _mm256_unpacklo_ps(row4, row5);
                    __m256 t5 = _mm256_unpackhi_ps(row4, row5);
                    __m256 t6 = _mm256_unpacklo_ps(row6, row7);
                    __m256 t7 = _mm256_unpackhi_ps(row6, row7);

                    row0 = _mm256_shuffle_ps(t0, t2, 0B01'00'01'00);    // 1 0 1 0
                    row1 = _mm256_shuffle_ps(t0, t2, 0B11'10'11'10);    // 3 2 3 2
                    row2 = _mm256_shuffle_ps(t1, t3, 0B01'00'01'00);
                    row3 = _mm256_shuffle_ps(t1, t3, 0B11'10'11'10);
                    row4 = _mm256_shuffle_ps(t4, t6, 0B01'00'01'00);
                    row5 = _mm256_shuffle_ps(t4, t6, 0B11'10'11'10);
                    row6 = _mm256_shuffle_ps(t5, t7, 0B01'00'01'00);
                    row7 = _mm256_shuffle_ps(t5, t7, 0B11'10'11'10);

                    t0 = _mm256_permute2f128_ps(row0, row4, 0x20);
                    t1 = _mm256_permute2f128_ps(row1, row5, 0x20);
                    t2 = _mm256_permute2f128_ps(row2, row6, 0x20);
                    t3 = _mm256_permute2f128_ps(row3, row7, 0x20);
                    t4 = _mm256_permute2f128_ps(row0, row4, 0x31);
                    t5 = _mm256_permute2f128_ps(row1, row5, 0x31);
                    t6 = _mm256_permute2f128_ps(row2, row6, 0x31);
                    t7 = _mm256_permute2f128_ps(row3, row7, 0x31);

                    float *dptr = (float*)input + N * j + i;
                    _mm256_storeu_ps(dptr, t0);
                    _mm256_storeu_ps(dptr += N, t1);
                    _mm256_storeu_ps(dptr += N, t2);
                    _mm256_storeu_ps(dptr += N, t3);
                    _mm256_storeu_ps(dptr += N, t4);
                    _mm256_storeu_ps(dptr += N, t5);
                    _mm256_storeu_ps(dptr += N, t6);
                    _mm256_storeu_ps(dptr += N, t7);

                    // PrintArr(M, output)
                }
            }
        }
    }

    // 计算剩余
    if(n != N)
    {
        for(i = 0, idxin = 0; i < m; ++i)
        {
            for(size_t j = n, idxout = n * N; j < N; ++j)
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