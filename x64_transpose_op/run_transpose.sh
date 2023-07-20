#!/bin/bash

if [ ! -d "bin" ]; then
  mkdir bin
fi

if [ ! -d "build" ]; then
  mkdir build
fi

g++ -c -O0 -g benchmark.cc -o ./build/benchmark.o
g++ -c -O0 -g transpose_v1_base.cc -o ./build/transpose_v1_base.o
g++ -c -O0 -g transpose_v2_cache.cc -o ./build/transpose_v2_cache.o
g++ -c -O0 -g -mavx transpose_v3_simd.cc -o ./build/transpose_v3_simd.o
g++ -o ./bin/benchmark ./build/*

./bin/benchmark