#ifndef MYTRANS_H_H
#define MYTRANS_H_H

#include <iostream>
#include <assert.h>
#include <immintrin.h>      // avx, 16 x 256bit regs

using std::cout;
using std::endl;
const size_t BLOCK = 64;    // cache_block_size = 64B
const size_t YMMLEN = 32;   // ymm_register_size = 256bit = 32B

void Transpose_v1_base(const float* input, float* output, size_t M, size_t N);
// void Transpose_v1_base_read(const float* input, float* output, size_t M, size_t N);
// void Transpose_v1_base_write(const float* input, float* output, size_t M, size_t N);
void Transpose_v2_cache(const float* input, float* output, size_t M, size_t N);
void Transpose_v3_simd(const float* input, float* output, size_t M, size_t N);

#endif