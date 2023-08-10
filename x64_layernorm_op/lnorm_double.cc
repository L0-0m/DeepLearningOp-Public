#include <immintrin.h>
#include <cassert>
#include <cmath>
#include <iostream>

#include "layernorm.h"

//
// algo: single pass with double precision
//
void LNORMOp::DoForwardDouble(float *output, const float *input, int batch_size, int seq_len) {

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

  for(int i = 0, idx = 0, lm = limit, lm_dim = limit_dim; i < data_len; ++idx) {
    int j, boundary = i + seq_x_dim;

    // compute mean, meansq
    __m256d sum_v = _mm256_setzero_pd();
    __m256d sumsq_v = _mm256_setzero_pd();

    for(j = i; j < lm; j += kPack) {
      __m256 v = _mm256_loadu_ps(input + j);

      __m128 v_1 = _mm256_castps256_ps128(v);
      __m128 v_2 = _mm256_extractf128_ps(v, 1);
      __m256d d_1 = _mm256_cvtps_pd(v_1);
      __m256d d_2 = _mm256_cvtps_pd(v_2);

      sum_v = _mm256_add_pd(sum_v, d_1);
      sum_v = _mm256_add_pd(sum_v, d_2);
      sumsq_v = _mm256_fmadd_pd(d_1, d_1, sumsq_v);
      sumsq_v = _mm256_fmadd_pd(d_2, d_2, sumsq_v);
    }

    if(j < boundary) {
      __m256 v = _mm256_maskload_ps(input + j, mask_v);

      __m128 v_1 = _mm256_castps256_ps128(v);
      __m128 v_2 = _mm256_extractf128_ps(v, 1);
      __m256d d_1 = _mm256_cvtps_pd(v_1);
      __m256d d_2 = _mm256_cvtps_pd(v_2);

      sum_v = _mm256_add_pd(sum_v, d_1);
      sum_v = _mm256_add_pd(sum_v, d_2);
      sumsq_v = _mm256_fmadd_pd(d_1, d_1, sumsq_v);
      sumsq_v = _mm256_fmadd_pd(d_2, d_2, sumsq_v);
    }

    _mm256_store_pd(tmp8d1_buffer_, sum_v);
    _mm256_store_pd(tmp8d2_buffer_, sumsq_v);

    double sum = 0.0, sumsq = 0.0;
    for(int k = 0; k < 4; ++k) {
      sum += tmp8d1_buffer_[k];
      sumsq += tmp8d2_buffer_[k];
    }

    double mean = sum / seq_x_dim;
    double std = std::sqrt(sumsq / seq_x_dim - mean * mean + 1e-5);
    __m256 mean_v = _mm256_set1_ps(mean);
    __m256 std_v = _mm256_set1_ps(std);

    // compute norm
    for(int j = i; j < boundary; ) {

      int k, idx_dim, boundary_k = j + input_size_;

      for(k = j, idx_dim = 0; k < lm_dim; k += kPack, idx_dim += kPack) {
        __m256 x = _mm256_loadu_ps(input + k);
        __m256 gm = _mm256_load_ps(weights_.w_gm + idx_dim);
        __m256 bi = _mm256_load_ps(weights_.w_bi + idx_dim);
        x = _mm256_sub_ps(x, mean_v);
        x = _mm256_div_ps(x, std_v);
        x = _mm256_fmadd_ps(x, gm, bi);
        _mm256_storeu_ps(output + k, x);
      }

      if(k < boundary_k) {
        __m256 x = _mm256_maskload_ps(input + k, mask_dim);
        __m256 gm = _mm256_load_ps(weights_.w_gm + idx_dim);
        __m256 bi = _mm256_load_ps(weights_.w_bi + idx_dim);
        x = _mm256_sub_ps(x, mean_v);
        x = _mm256_div_ps(x, std_v);
        x = _mm256_fmadd_ps(x, gm, bi);
        _mm256_maskstore_ps(output + k, mask_dim, x);
      }

      j = boundary_k, lm_dim += input_size_;
    }

    i = boundary, lm += seq_x_dim;
  }
}
