#include <immintrin.h>
#include <cassert>
#include <iostream>
#include <cmath>

#include "layernorm.h"

void LNORMOp::DoForwardNaive(float *output, const float *input, int batch_size, int seq_len) {

  constexpr int kPack = 8;
  const int seq_x_dim = seq_len * input_size_;
  const int data_len = seq_x_dim * batch_size;
  
  int nleft = seq_x_dim % kPack;
  int limit = nleft == 0 ? seq_x_dim : seq_x_dim - nleft;

  int nleft_batch = batch_size % kPack;
  int limit_batch = nleft_batch == 0 ? batch_size : batch_size - nleft_batch;

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

  // compute mean
  for(int i = 0, idx = 0, lm = limit; i < data_len; ++idx) {
    int j, boundary = i + seq_x_dim;
    
    __m256 sum_v = _mm256_setzero_ps();
    for(j = i; j < lm; j += kPack) {
      __m256 v = _mm256_loadu_ps(input + j);
      sum_v = _mm256_add_ps(sum_v, v);
    }

    if(j < boundary) {
      __m256 v = _mm256_maskload_ps(input + j, mask_v);
      sum_v = _mm256_add_ps(sum_v, v);
    }

    _mm256_store_ps(tmp8f1_buffer_, sum_v);
    float sum = 0.0f;
    for(int k = 0; k < 8; ++k) {
      sum += tmp8f1_buffer_[k];
    }

    batch_mean_buffer_[idx] = sum / seq_x_dim;
    i = boundary, lm += seq_x_dim;
  }

  // compute x_i - mean， var
  for(int i = 0, idx = 0, lm = limit; i < data_len; ++idx) {
    int j, boundary = i + seq_x_dim;
    
    float mean = batch_mean_buffer_[idx];
    __m256 mean_v = _mm256_set1_ps(mean);
    __m256 sum_v = _mm256_setzero_ps();
    for(j = i; j < lm; j += kPack) {
      __m256 v = _mm256_loadu_ps(input + j);
      v = _mm256_sub_ps(v, mean_v);
      _mm256_storeu_ps(x_sub_mean_buffer_ + j, v);
      sum_v = _mm256_fmadd_ps(v, v, sum_v);
    }
    
    if(j < boundary) {
      __m256 v = _mm256_maskload_ps(input + j, mask_v);
      v = _mm256_sub_ps(v, mean_v);
      _mm256_maskstore_ps(x_sub_mean_buffer_ + j, mask_v, v);
      sum_v = _mm256_fmadd_ps(v, v, sum_v);
    }
    _mm256_store_ps(tmp8f1_buffer_, sum_v);
    
    float sum = 0.0f, t;
    for(int k = 0; k < 8; ++k) {
      sum += tmp8f1_buffer_[k];
    }

    batch_mean_buffer_[idx] = sum / seq_x_dim + 1e-5f;
    i = boundary, lm += seq_x_dim;
  }

  switch(nleft_batch) {
      case 1: mask_v = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, -1); break;
      case 2: mask_v = _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1); break;
      case 3: mask_v = _mm256_set_epi32(0, 0, 0, 0, 0, -1, -1, -1); break;
      case 4: mask_v = _mm256_set_epi32(0, 0, 0, 0, -1, -1, -1, -1); break;
      case 5: mask_v = _mm256_set_epi32(0, 0, 0, -1, -1, -1, -1, -1); break;
      case 6: mask_v = _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1); break;
      case 7: mask_v = _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1); break;
  }
  {
    int i, lm = limit_batch;
    for(i = 0; i < lm; i += kPack) {
      __m256 v = _mm256_load_ps(batch_var_buffer_ + i);
      v = _mm256_sqrt_ps(v);
      _mm256_store_ps(batch_var_buffer_ + i, v);
    }
    if(i < batch_size) {
      __m256 v = _mm256_maskload_ps(batch_var_buffer_ + i, mask_v);
      v = _mm256_sqrt_ps(v);
      _mm256_maskstore_ps(batch_var_buffer_ + i, mask_v, v);
    }
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
  for(int i = 0, idx_std = 0, lm = limit_dim; i < data_len; ++ idx_std) {
    int boundary_j = i + seq_x_dim;
    
    for(int j = i; j < boundary_j;) {

      int k, idx, boundary_k = j + input_size_;

      float std = batch_var_buffer_[idx_std];
      __m256 stdx4 = _mm256_set1_ps(std);
      for(k = j, idx = 0; k < lm; k += kPack, idx += kPack) {
        __m256 v = _mm256_loadu_ps(x_sub_mean_buffer_ + k);
        __m256 gm = _mm256_load_ps(weights_.w_gm + idx);
        __m256 bi = _mm256_load_ps(weights_.w_bi + idx);
        v = _mm256_div_ps(v, stdx4);
        v = _mm256_fmadd_ps(v, gm, bi);
        _mm256_storeu_ps(output + k, v);
      }

      if(k < boundary_k) {
        __m256 v = _mm256_maskload_ps(x_sub_mean_buffer_ + k, mask_v);
        __m256 gm = _mm256_load_ps(weights_.w_gm + idx);
        __m256 bi = _mm256_load_ps(weights_.w_bi + idx);
        v = _mm256_div_ps(v, stdx4);
        v = _mm256_fmadd_ps(v, gm, bi);
        _mm256_maskstore_ps(output + k, mask_v, v);
      }

      j = boundary_k, lm += input_size_;
    }

    i = boundary_j;
  }
}