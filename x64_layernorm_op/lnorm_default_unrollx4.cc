#include <immintrin.h>
#include <cassert>
#include <cmath>
#include <iostream>

#include "layernorm.h"

//
// algo: single pass unrolling x 4 to ultilize registers
//
void LNORMOp::DoForwardDefault(float *output, const float *input, int batch_size, int seq_len) {

  constexpr int kPack = 8;
  constexpr int kPackx4 = kPack * 4;
  const int seq_x_dim = seq_len * input_size_;
  const int data_len = seq_x_dim * batch_size;
  
  int nleft = seq_x_dim % kPack;
  int limit = nleft == 0 ? seq_x_dim : seq_x_dim - nleft;

  int nleft_unroll = limit % kPackx4;
  int limit_unroll = nleft_unroll == 0 ? limit : limit - nleft_unroll;

  int nleft_dim = input_size_ % kPack;
  int limit_dim = nleft_dim == 0 ? input_size_ : input_size_ - nleft_dim;

  int nleft_dim_unroll = limit_dim % kPackx4;
  int limit_dim_unroll = nleft_dim_unroll == 0 ? limit_dim : limit_dim - nleft_dim_unroll;

  __m256i mask_v;
  switch(nleft) {
    case 1: mask_v = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, -1); break;
    case 2: mask_v = _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1); break;
    case 3: mask_v = _mm256_set_epi32(0, 0, 0, 0, 0, -1, -1, -1); break;
    case 4: mask_v = _mm256_set_epi32(0, 0, 0, 0, -1, -1, -1, -1); break;
    case 5: mask_v = _mm256_set_epi32(0, 0, 0, -1, -1, -1, -1, -1); break;
    case 6: mask_v = _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1); break;
    case 7: mask_v = _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1); break;
  }

  __m256i mask_dim;
  switch(nleft_dim) {
    case 1: mask_dim = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, -1); break;
    case 2: mask_dim = _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1); break;
    case 3: mask_dim = _mm256_set_epi32(0, 0, 0, 0, 0, -1, -1, -1); break;
    case 4: mask_dim = _mm256_set_epi32(0, 0, 0, 0, -1, -1, -1, -1); break;
    case 5: mask_dim = _mm256_set_epi32(0, 0, 0, -1, -1, -1, -1, -1); break;
    case 6: mask_dim = _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1); break;
    case 7: mask_dim = _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1); break;
  }
  
  for(int i = 0, lm = limit, lmx4 = limit_unroll, lm_dim = limit_dim, lmx4_dim = limit_dim_unroll; i < data_len;) {
    int j, boundary = i + seq_x_dim;
    
    // compute mean, std
    __m256 sum_v = _mm256_setzero_ps();
    __m256 sumsq_v = _mm256_setzero_ps();

    for(j = i; j < lmx4;) {
      __m256 v = _mm256_loadu_ps(input + j);
      j += kPack;
      sum_v = _mm256_add_ps(sum_v, v);
      sumsq_v = _mm256_fmadd_ps(v, v, sumsq_v);
      
      __m256 v = _mm256_loadu_ps(input + j);
      j += kPack;
      sum_v = _mm256_add_ps(sum_v, v);
      sumsq_v = _mm256_fmadd_ps(v, v, sumsq_v);

      __m256 v = _mm256_loadu_ps(input + j);
      j += kPack;
      sum_v = _mm256_add_ps(sum_v, v);
      sumsq_v = _mm256_fmadd_ps(v, v, sumsq_v);

      __m256 v = _mm256_loadu_ps(input + j);
      j += kPack;
      sum_v = _mm256_add_ps(sum_v, v);
      sumsq_v = _mm256_fmadd_ps(v, v, sumsq_v);
    }

    for(; j < lm; j += kPack) {
      __m256 v = _mm256_loadu_ps(input + j);
      sum_v = _mm256_add_ps(sum_v, v);
      sumsq_v = _mm256_fmadd_ps(v, v, sumsq_v);
    }

    if(j < boundary) {
      __m256 v = _mm256_maskload_ps(input + j, mask_v);
      sum_v = _mm256_add_ps(sum_v, v);
      sumsq_v = _mm256_fmadd_ps(v, v, sumsq_v);
    }

    _mm256_store_ps(tmp8f1_buffer_, sum_v);
    _mm256_store_ps(tmp8f2_buffer_, sumsq_v);
    float sum = 0.0f, sumsq = 0.0f;
    for(int k = 0; k < kPack; ++k) {
      sum += tmp8f1_buffer_[k];
      sumsq += tmp8f2_buffer_[k];
    }

    float mean = sum / seq_x_dim;
    float std = std::sqrt(sumsq - mean * mean + 1e-5f);

    // affine
    __m256 mean_v = _mm256_set1_ps(mean);
    __m256 std_v = _mm256_set1_ps(std);
    for(int j = i; j < boundary;) {

      int k, idx, boundary_k = j + input_size_;

      for(k = j, idx = 0; k < lmx4_dim;) {
        __m256 v = _mm256_loadu_ps(input + k);
        __m256 gm = _mm256_load_ps(weights_.w_gm + idx);
        __m256 bi = _mm256_load_ps(weights_.w_bi + idx);
        k += kPack, idx += kPack;
        v = _mm256_sub_ps(v, mean_v);
        v = _mm256_div_ps(v, std_v);
        v = _mm256_fmadd_ps(v, gm, bi);
        _mm256_storeu_ps(output + k, v);
        
        __m256 v = _mm256_loadu_ps(input + k);
        __m256 gm = _mm256_load_ps(weights_.w_gm + idx);
        __m256 bi = _mm256_load_ps(weights_.w_bi + idx);
        k += kPack, idx += kPack;
        v = _mm256_sub_ps(v, mean_v);
        v = _mm256_div_ps(v, std_v);
        v = _mm256_fmadd_ps(v, gm, bi);
        _mm256_storeu_ps(output + k, v);

        __m256 v = _mm256_loadu_ps(input + k);
        __m256 gm = _mm256_load_ps(weights_.w_gm + idx);
        __m256 bi = _mm256_load_ps(weights_.w_bi + idx);
        k += kPack, idx += kPack;
        v = _mm256_sub_ps(v, mean_v);
        v = _mm256_div_ps(v, std_v);
        v = _mm256_fmadd_ps(v, gm, bi);
        _mm256_storeu_ps(output + k, v);

        __m256 v = _mm256_loadu_ps(input + k);
        __m256 gm = _mm256_load_ps(weights_.w_gm + idx);
        __m256 bi = _mm256_load_ps(weights_.w_bi + idx);
        k += kPack, idx += kPack;
        v = _mm256_sub_ps(v, mean_v);
        v = _mm256_div_ps(v, std_v);
        v = _mm256_fmadd_ps(v, gm, bi);
        _mm256_storeu_ps(output + k, v);
      }

      for(k = j, idx = 0; k < lm_dim; k += kPack, idx += kPack) {
        __m256 v = _mm256_loadu_ps(input + k);
        __m256 gm = _mm256_load_ps(weights_.w_gm + idx);
        __m256 bi = _mm256_load_ps(weights_.w_bi + idx);
        v = _mm256_sub_ps(v, mean_v);
        v = _mm256_div_ps(v, std_v);
        v = _mm256_fmadd_ps(v, gm, bi);
        _mm256_storeu_ps(output + k, v);
      }

      if(k < boundary_k) {
        __m256 v = _mm256_maskload_ps(input + k, mask_dim);
        __m256 gm = _mm256_load_ps(weights_.w_gm + idx);
        __m256 bi = _mm256_load_ps(weights_.w_bi + idx);
        v = _mm256_sub_ps(v, mean_v);
        v = _mm256_div_ps(v, std_v);
        v = _mm256_fmadd_ps(v, gm, bi);
        _mm256_maskstore_ps(output + k, mask_dim, v);
      }

      j = boundary_k, lm_dim += input_size_, lmx4_dim += input_size_;
    }

    i = boundary, lm += seq_x_dim, lmx4 += seq_x_dim;
  }
}
