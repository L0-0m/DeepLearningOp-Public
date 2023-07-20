#pragma once

#include <cstddef>
#include <cstdint>

enum class LNORMAlgo : int32_t { Default = 0, Naive = 1, Double = 3, SubK = 4 };

struct LNORMWeights {
  LNORMWeights() {}

  LNORMWeights(float* wgm, bool pack_wgm, float* wbi, bool pack_wbi)
      : w_gm(wgm), pack_w_ih(pack_wgm), w_bi(wbi), pack_w_bi(pack_wbi) {}

  float* w_gm = nullptr;
  bool pack_w_ih = false;

  float* w_bi = nullptr;
  bool pack_w_bi = false;
};

class LNORMOp {
 public:
  LNORMOp(int input_size, int max_batch_size, int max_batch_x_seq_size, // TODO Add params for selecting normalization dim
          LNORMAlgo algo);                                              // Add params 'is_affine'
  virtual ~LNORMOp();

  void SetWeights(const LNORMWeights weights);

  // units: sizeof(float)
  size_t GetBufferSize();

  void SetBuffer(const void *buffer);

  void DoForward(float *output, const float *input, int batch_size, int seq_len);

 private:
  void (LNORMOp::*DoForwardDirection)(float *output, const float *input, int batch_size, int seq_len);

  void DoForwardDefault(float *output, const float *input, int batch_size, int seq_len);

  void DoForwardNaive(float *output, const float *input, int batch_size, int seq_len);

  void DoForwardDouble(float *output, const float *input, int batch_size, int seq_len);

  void DoForwardSubK(float *output, const float *input, int batch_size, int seq_len);

 private:
  LNORMAlgo algo_;

  // bool is_affine;

  int input_size_;  // input size = proj size

  float *buffer;

  // weights
  LNORMWeights weights_;

  // buffer size 
  size_t buffer_size_;
  size_t batch_mean_size_;    // max batch size
  size_t x_sub_mean_size_;    // max data size

  // aligned memory
  float *batch_mean_buffer_ = nullptr;
  float *batch_var_buffer_ = nullptr;
  float *x_sub_mean_buffer_ = nullptr;
  float *tmp8f1_buffer_ = nullptr;
  float *tmp8f2_buffer_ = nullptr;
  double *tmp8d1_buffer_ = nullptr;
  double *tmp8d2_buffer_ = nullptr;

  bool is_set_weight_ = false;
  bool is_set_buffer_ = false;
};
