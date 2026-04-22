#pragma once

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <memory>
#include <vector>

#include "external_depth_sim/cuda_se3.cuh"

namespace external_depth_sim {

struct CameraParams {
  float fx = 80.0f;
  float fy = 80.0f;
  float cx = 80.0f;
  float cy = 45.0f;
  int image_width = 160;
  int image_height = 90;
  float min_depth_dist = 0.1f;
  float max_depth_dist = 20.0f;
};

struct Vector3f {
  float x, y, z;
  __device__ __host__ Vector3f() : x(0.0f), y(0.0f), z(0.0f) {}
  __device__ __host__ Vector3f(float x_val, float y_val, float z_val) : x(x_val), y(y_val), z(z_val) {}
};

struct Vector3i {
  int x, y, z;
  __device__ __host__ Vector3i() : x(0), y(0), z(0) {}
  __device__ __host__ Vector3i(int x_val, int y_val, int z_val) : x(x_val), y(y_val), z(z_val) {}
};

struct GridMapView {
  int* map_cuda_ = nullptr;
  float raycast_step_ = 0.1f;
  float resolution_ = 0.1f;
  float origin_x_ = 0.0f, origin_y_ = 0.0f, origin_z_ = 0.0f;
  int grid_size_x_ = 0, grid_size_y_ = 0, grid_size_z_ = 0, grid_size_yz_ = 0;
  int occupy_threshold_ = 1;

  __host__ __device__ Vector3i Pos2Vox(const Vector3f& pos) const {
    return Vector3i(
        static_cast<int>(floor((pos.x - origin_x_) / resolution_)),
        static_cast<int>(floor((pos.y - origin_y_) / resolution_)),
        static_cast<int>(floor((pos.z - origin_z_) / resolution_)));
  }

  __host__ __device__ Vector3f Vox2Pos(const Vector3i& vox) const {
    return Vector3f(
        (vox.x + 0.5f) * resolution_ + origin_x_,
        (vox.y + 0.5f) * resolution_ + origin_y_,
        (vox.z + 0.5f) * resolution_ + origin_z_);
  }

  __host__ __device__ int Vox2Idx(const Vector3i& vox) const {
    return vox.x * grid_size_yz_ + vox.y * grid_size_z_ + vox.z;
  }

  __device__ int mapQuery(const Vector3f& pos) const {
    Vector3i vox = Pos2Vox(pos);
    if (vox.x < 0 || vox.y < 0 || vox.z < 0 || vox.x >= grid_size_x_ || vox.y >= grid_size_y_ || vox.z >= grid_size_z_) {
      return 0;
    }
    return map_cuda_[Vox2Idx(vox)] > occupy_threshold_ ? 1 : 0;
  }
};

class GridMap {
 public:
  GridMap();
  ~GridMap();
  GridMap(const GridMap&) = delete;
  GridMap& operator=(const GridMap&) = delete;
  GridMap(GridMap&& other) noexcept;
  GridMap& operator=(GridMap&& other) noexcept;

  bool buildFromCloud(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud, float resolution, int occupy_threshold);
  bool valid() const;
  void freeGridMap();
  GridMapView deviceView() const;

  __host__ __device__ Vector3i Pos2Vox(const Vector3f& pos) const;
  __host__ __device__ Vector3f Vox2Pos(const Vector3i& vox) const;
  __host__ __device__ int Vox2Idx(const Vector3i& vox) const;
  __device__ int mapQuery(const Vector3f& pos) const;

  float raycast_step_;

 private:
  int* map_cuda_;
  float resolution_;
  float origin_x_, origin_y_, origin_z_;
  int grid_size_x_, grid_size_y_, grid_size_z_, grid_size_yz_;
  int occupy_threshold_;
};

void renderDepthImage(const GridMap& grid_map, const CameraParams& camera_param, const SE3<float>& T_wc, cv::Mat& depth_image);

}