#pragma once
#include <cuda_runtime.h>

#ifndef __CUDACC__
#include <math.h>
inline float fminf(float a, float b) { return a < b ? a : b; }
inline float fmaxf(float a, float b) { return a > b ? a : b; }
inline float rsqrtf(float x) { return 1.0f / sqrtf(x); }
#endif
