#pragma once
#include "external_depth_sim/cuda_matrix.cuh"
#include "external_depth_sim/cuda_toolkit_helper_math.h"

namespace external_depth_sim {
template<typename Type>
struct SE3 {
  __host__ __device__ __forceinline__ SE3() {
    data(0,0)=1; data(0,1)=0; data(0,2)=0; data(0,3)=0;
    data(1,0)=0; data(1,1)=1; data(1,2)=0; data(1,3)=0;
    data(2,0)=0; data(2,1)=0; data(2,2)=1; data(2,3)=0;
  }
  __host__ __device__ __forceinline__ SE3(Type qw, Type qx, Type qy, Type qz, Type tx, Type ty, Type tz) {
    const Type x = 2 * qx, y = 2 * qy, z = 2 * qz;
    const Type wx = x * qw, wy = y * qw, wz = z * qw;
    const Type xx = x * qx, xy = y * qx, xz = z * qx, yy = y * qy, yz = z * qy, zz = z * qz;
    data(0,0)=1-(yy+zz); data(0,1)=xy-wz;   data(0,2)=xz+wy;   data(0,3)=tx;
    data(1,0)=xy+wz;   data(1,1)=1-(xx+zz); data(1,2)=yz-wx;   data(1,3)=ty;
    data(2,0)=xz-wy;   data(2,1)=yz+wx;   data(2,2)=1-(xx+yy); data(2,3)=tz;
  }
  __host__ __device__ __forceinline__ SE3<Type> inv() const {
    SE3<Type> result;
    result.data[0]=data[0]; result.data[1]=data[4]; result.data[2]=data[8];
    result.data[4]=data[1]; result.data[5]=data[5]; result.data[6]=data[9];
    result.data[8]=data[2]; result.data[9]=data[6]; result.data[10]=data[10];
    result.data[3] = -data[0]*data[3] - data[4]*data[7] - data[8]*data[11];
    result.data[7] = -data[1]*data[3] - data[5]*data[7] - data[9]*data[11];
    result.data[11]= -data[2]*data[3] - data[6]*data[7] - data[10]*data[11];
    return result;
  }
  Matrix<Type, 3, 4> data;
};

template<typename Type>
__host__ __device__ __forceinline__ float3 operator*(const SE3<Type> &se3, const float3 &p) {
  return make_float3(
    se3.data(0,0)*p.x + se3.data(0,1)*p.y + se3.data(0,2)*p.z + se3.data(0,3),
    se3.data(1,0)*p.x + se3.data(1,1)*p.y + se3.data(1,2)*p.z + se3.data(1,3),
    se3.data(2,0)*p.x + se3.data(2,1)*p.y + se3.data(2,2)*p.z + se3.data(2,3)
  );
}
}