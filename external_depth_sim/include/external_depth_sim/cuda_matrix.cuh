#pragma once
#include <cuda_runtime.h>
#include <ostream>
#include <iomanip>

namespace external_depth_sim {
template<typename Type, unsigned R, unsigned C>
struct Matrix {
  __host__ __device__ __forceinline__ Type operator()(int row, int col) const { return data[row * C + col]; }
  __host__ __device__ __forceinline__ Type &operator()(int row, int col) { return data[row * C + col]; }
  __host__ __device__ __forceinline__ Type operator[](int ind) const { return data[ind]; }
  __host__ __device__ __forceinline__ Type &operator[](int ind) { return data[ind]; }
  Type data[R * C];
};
}