#include <immintrin.h>
#include <cassert>
#include <cmath>
#include <iostream>

#include "layernorm.h"

void LNORMOp::DoForwardDefault(float *output, const float *input, int batch_size, int seq_len) {

  constexpr int kPack = 8;
  const int seq_x_dim = seq_len * input_size_;
  const int data_len = seq_x_dim * batch_size;
  
  int nleft = seq_x_dim % kPack;
  int limit = nleft == 0 ? seq_x_dim : seq_x_dim - nleft;

  int nleft_dim = input_size_ % kPack;
  int limit_dim = nleft_dim == 0 ? input_size_ : input_size_ - nleft_dim;

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

  // compute mean, meansq
  for(int i = 0, idx = 0, lm = limit; i < data_len; ++idx) {
    int j, boundary = i + seq_x_dim;

    __m256 sum_v = _mm256_setzero_ps();
    __m256 sumsq_v = _mm256_setzero_ps();
    for(j = i; j < lm; j += kPack) {
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

    float t = sum / seq_x_dim;
    batch_mean_buffer_[idx] = t;
    batch_var_buffer_[idx] = std::sqrt(sumsq / seq_x_dim - t * t + 1e-5f);

    i = boundary, lm += seq_x_dim;
  }

  switch(nleft_dim) {
      case 1: mask_v = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, -1); break;
      case 2: mask_v = _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1); break;
      case 3: mask_v = _mm256_set_epi32(0, 0, 0, 0, 0, -1, -1, -1); break;
      case 4: mask_v = _mm256_set_epi32(0, 0, 0, 0, -1, -1, -1, -1); break;
      case 5: mask_v = _mm256_set_epi32(0, 0, 0, -1, -1, -1, -1, -1); break;
      case 6: mask_v = _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1); break;
      case 7: mask_v = _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1); break;
  }

  // compute norm
  for(int i = 0, lm = limit_dim, idx = 0; i < data_len; ++idx) {
    int boundary_j = i + seq_x_dim;

    float mean = batch_mean_buffer_[idx];
    float std = batch_var_buffer_[idx];

    __m256 mean_v = _mm256_set1_ps(mean);
    __m256 std_v = _mm256_set1_ps(std);

    for(int j = i; j < boundary_j; ) {

      int k, idx_dim, boundary_k = j + input_size_;

      for(k = j, idx_dim = 0; k < lm; k += kPack, idx_dim += kPack) {
        __m256 x = _mm256_loadu_ps(input + k);
        __m256 gm = _mm256_load_ps(weights_.w_gm + idx_dim);
        __m256 bi = _mm256_load_ps(weights_.w_bi + idx_dim);
        x = _mm256_sub_ps(x, mean_v);
        x = _mm256_div_ps(x, std_v);
        x = _mm256_fmadd_ps(x, gm, bi);
        _mm256_storeu_ps(output + k, x);
      }

      if(k < boundary_k) {
        __m256 x = _mm256_maskload_ps(input + k, mask_v);
        __m256 gm = _mm256_load_ps(weights_.w_gm + idx_dim);
        __m256 bi = _mm256_load_ps(weights_.w_bi + idx_dim);
        x = _mm256_sub_ps(x, mean_v);
        x = _mm256_div_ps(x, std_v);
        x = _mm256_fmadd_ps(x, gm, bi);
        _mm256_maskstore_ps(output + k, mask_v, x);
      }

      j = boundary_k, lm += input_size_;
    }

    i = boundary_j;
  }

}
