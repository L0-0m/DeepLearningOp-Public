#include <cstring>
#include <iostream>
#include <memory>
#include<ctime>

#include "layernorm.h"
#include "alignment.h"

void Benchmark(const LNORMAlgo algo) {
  const int batch_size = 114;
  const int seq_len = 1069;
  const int input_size = 128;
  // const int batch_size = 2138;
  // const int seq_len = 698;
  // const int input_size = 255;

  const int max_batch_size = batch_size + 10;
  const int max_batch_x_seq_size = batch_size * seq_len + 100;

  auto lnorm_op = std::make_unique<LNORMOp>(input_size, max_batch_size, max_batch_x_seq_size, algo);

  // lnorm weights
  const int alignment = 32;

  size_t w_gm_size = sizeof(float) * input_size;
  float* w_gm = (float*)AlignedMalloc(alignment, w_gm_size);
  std::memset(w_gm, -1, w_gm_size);

  size_t w_bi_size = sizeof(float) * input_size;
  float* w_bi = (float*)AlignedMalloc(alignment, w_bi_size);
  std::memset(w_bi, 0, w_bi_size);

  bool pack_w_gm = true;
  bool pack_w_bi = true;

  const LNORMWeights weights = {w_gm, pack_w_gm, w_bi, pack_w_bi};

  lnorm_op->SetWeights(weights);

  // set buffer
  float* buffer = (float*)AlignedMalloc(alignment, sizeof(float) * lnorm_op->GetBufferSize());
  lnorm_op->SetBuffer(buffer);

  // malloc input and output
  size_t input_data_size = batch_size * seq_len * input_size;
  float *input = (float*)malloc(sizeof(float) * input_data_size);   // use undefined value
  float *output = (float*)malloc(sizeof(float) * input_data_size);

  clock_t start, end;

  // warm up
  lnorm_op->DoForward(output, input, batch_size, seq_len);

  constexpr int kRepeats = 10;
  float time[kRepeats] = {0};

  for (int i = 0; i < kRepeats; ++i) {
    start = clock();
    lnorm_op->DoForward(output, input, batch_size, seq_len);
    end = clock();
    time[i] = ((float)end-start)/CLOCKS_PER_SEC;
  }

if (algo == LNORMAlgo::Default)
    std::cout << "lnorm time (s): default algorithm" << std::endl;
else if(algo == LNORMAlgo::Naive)
    std::cout << "lnorm time (s): naive algorithm" << std::endl;
else if(algo == LNORMAlgo::Double)
    std::cout << "lnorm time (s): double algorithm" << std::endl;
else
    std::cout << "lnorm time (s): subk algorithm" << std::endl;


  double avg = 0;
  for (int i = 0; i < kRepeats - 1; ++i) {
    std::cout << time[i] << ", ";
    avg += time[i];
  }
  std::cout << time[kRepeats - 1] << std::endl;
  avg = (avg + time[kRepeats - 1]) / kRepeats;
  std::cout << "avg time: " << avg << std::endl;

  // free memory
  AlignedFree(w_gm);
  AlignedFree(w_bi);
  AlignedFree(buffer);
  free(input);
  free(output);
}

int main() {
  // Benchmark(LNORMAlgo::Naive);
  // Benchmark(LNORMAlgo::Double);
  // Benchmark(LNORMAlgo::Default);
  Benchmark(LNORMAlgo::SubK);

  return 0;
}
