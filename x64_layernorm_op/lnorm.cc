#include <cassert>
#include <iostream>

#include "alignment.h"
#include "layernorm.h"

LNORMOp::LNORMOp(int input_size, int max_batch_size, int max_batch_x_seq_size, 
                 LNORMAlgo algo = LNORMAlgo::Default)
    : input_size_(input_size),
      algo_(algo),
      DoForwardDirection(nullptr) {

  batch_mean_size_ = GetAlignSize<float>(max_batch_size);
  x_sub_mean_size_ = GetAlignSize<float>(input_size_ * max_batch_x_seq_size);

  if (algo == LNORMAlgo::Default) { // single
    buffer_size_ = GetAlignSize<float>(batch_mean_size_ * 2 + 8 * 2);

    DoForwardDirection = &LNORMOp::DoForwardDefault;

  } else if (algo == LNORMAlgo::Naive) {
    buffer_size_ = GetAlignSize<float>(batch_mean_size_ + 8 + x_sub_mean_size_);

    DoForwardDirection = &LNORMOp::DoForwardNaive;
  } else if (algo == LNORMAlgo::Double) {
    buffer_size_ = GetAlignSize<float>(batch_mean_size_ * 2 + 8 * 2 * 2);

    DoForwardDirection = &LNORMOp::DoForwardDouble;
  } else if (algo == LNORMAlgo::SubK) {
    buffer_size_ = GetAlignSize<float>(batch_mean_size_ * 2 + 8 * 2);

    DoForwardDirection = &LNORMOp::DoForwardSubK;
  } else {
    // throw exception
  }

//   if (is_affine)
//     
//   else
//     
}

LNORMOp::~LNORMOp() {}

void LNORMOp::SetWeights(const LNORMWeights weights) {
  weights_ = weights;

  is_set_weight_ = true;
}

size_t LNORMOp::GetBufferSize() { return buffer_size_; }

void LNORMOp::SetBuffer(const void* buffer) {
  assert((((size_t)buffer & 31) == 0) && "Should set aligned buffer!");
  tmp8f1_buffer_ = reinterpret_cast<float*>(const_cast<void*>(buffer));

  batch_mean_buffer_ = tmp8f1_buffer_ + 8;

  if (algo_ == LNORMAlgo::Default) {    // single pass
    tmp8f2_buffer_ = batch_mean_buffer_ + batch_mean_size_;
    batch_var_buffer_ = tmp8f2_buffer_ + 8;

  } else if (algo_ == LNORMAlgo::Naive) {
    x_sub_mean_buffer_ = batch_mean_buffer_ + batch_mean_size_;
    batch_var_buffer_ = batch_mean_buffer_;
    assert((x_sub_mean_buffer_ <= tmp8f1_buffer_ + buffer_size_) && "invalid buffer!");

  } else if (algo_ == LNORMAlgo::Double) {
    tmp8d1_buffer_ = reinterpret_cast<double*>(const_cast<void*>(buffer));
    tmp8d2_buffer_ = tmp8d1_buffer_ + 8;
    batch_mean_buffer_ = (float *)(tmp8d2_buffer_ + 8);
    batch_var_buffer_ = batch_mean_buffer_ + batch_mean_size_;

  } else if (algo_ == LNORMAlgo::SubK) {
    tmp8f2_buffer_ = batch_mean_buffer_ + batch_mean_size_;
    batch_var_buffer_ = tmp8f2_buffer_ + 8;

  } else {
    // throw exception
  }

  is_set_buffer_ = true;
}

// TODO: support Parallel
void LNORMOp::DoForward(float *output, const float *input, int batch_size, int seq_len) {
  assert(is_set_weight_ && is_set_buffer_ && "Should set weight and buffer before DoForward!");

  (this->*DoForwardDirection)(output, input, batch_size, seq_len);
}
