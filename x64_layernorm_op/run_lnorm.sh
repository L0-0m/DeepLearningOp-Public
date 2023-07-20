#!/bin/bash

if [ ! -d "bin" ]; then
  mkdir bin
fi

if [ ! -d "build" ]; then
  mkdir build
fi

g++ -c -O0 -g lnorm.cc -o ./build/lnorm.o
g++ -c -O0 -g bench_lnorm.cc -o ./build/bench_lnorm.o
g++ -c -O0 -g -mavx -mavx2 -mfma lnorm_default.cc -o ./build/lnorm_default.o
g++ -c -O0 -g -mavx -mavx2 -mfma lnorm_double.cc -o ./build/lnorm_double.o
g++ -c -O0 -g -mavx -mavx2 -mfma lnorm_naive.cc -o ./build/lnorm_naive.o
g++ -c -O0 -g -mavx -mavx2 -mfma lnorm_subk.cc -o ./build/lnorm_subk.o
g++ -o ./bin/bench_lnorm ./build/*

# ./bin/bench_lnorm