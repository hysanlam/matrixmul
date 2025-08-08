# matrixmul

CUDA + C++ implementations of dense matrix multiplication with a simple build system and optional benchmarking. Includes both a CPU baseline and GPU kernels.

## Features
- Reference CPU implementation in C++
- CUDA GPU implementation(s) (naïve and/or tiled/shared-memory variants)
- Build with **Makefile** (no CMake required)
- Optional helper script for quick builds and runs

## Requirements
- CUDA Toolkit (nvcc) and a CUDA-capable GPU
- C++17 compiler (g++ or clang++)
- make

## Build

Clean and compile both CPU and GPU versions:

```bash
make clean && make
