#include "mytranspose.h"

/*
 * baseline版本（读优先）。 
*/

void Transpose_v1_base(const float* input, float* output, size_t M, size_t N)
{
    size_t idxin = 0, idxout = 0;
    for(size_t i = 0; i < M; ++i)
    {
        idxout = 0;
        for(size_t j = 0; j < N; ++j)
        {
            output[idxout + i] = input[idxin++];
            idxout += N;
        }
    }
}

// 写优先
// void Transpose_v1_base_write(const float* input, float* output, size_t M, size_t N)
// {
//     size_t idxin = 0, idxout = 0;
//     for(size_t i = 0; i < M; ++i)
//     {
//         idxin = 0;
//         for(size_t j = 0; j < N; ++j)
//         {
//             output[idxout++] = input[idxin + i];
//             idxin += N;
//         }
//     }
// }
