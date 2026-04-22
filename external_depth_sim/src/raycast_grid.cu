#include "external_depth_sim/raycast_grid.hpp"

#include <Eigen/Core>
#include <pcl/common/common.h>

namespace external_depth_sim {

GridMap::GridMap()
    : raycast_step_(0.1f),
      map_cuda_(nullptr),
      resolution_(0.1f),
      origin_x_(0.0f), origin_y_(0.0f), origin_z_(0.0f),
      grid_size_x_(0), grid_size_y_(0), grid_size_z_(0), grid_size_yz_(0),
      occupy_threshold_(0) {}

GridMap::~GridMap() { freeGridMap(); }

GridMap::GridMap(GridMap&& other) noexcept { *this = std::move(other); }

GridMap& GridMap::operator=(GridMap&& other) noexcept {
  if (this != &other) {
    freeGridMap();
    map_cuda_ = other.map_cuda_;
    resolution_ = other.resolution_;
    origin_x_ = other.origin_x_;
    origin_y_ = other.origin_y_;
    origin_z_ = other.origin_z_;
    grid_size_x_ = other.grid_size_x_;
    grid_size_y_ = other.grid_size_y_;
    grid_size_z_ = other.grid_size_z_;
    grid_size_yz_ = other.grid_size_yz_;
    occupy_threshold_ = other.occupy_threshold_;
    raycast_step_ = other.raycast_step_;
    other.map_cuda_ = nullptr;
    other.grid_size_x_ = other.grid_size_y_ = other.grid_size_z_ = other.grid_size_yz_ = 0;
  }
  return *this;
}

bool GridMap::buildFromCloud(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud, float resolution, int occupy_threshold) {
  freeGridMap();
  if (!cloud || cloud->points.empty()) {
    return false;
  }

  const float epsilon = 0.001f;
  Eigen::Vector4f min_pt, max_pt;
  pcl::getMinMax3D(*cloud, min_pt, max_pt);
  const float length = max_pt(0) - min_pt(0) + 2 * epsilon;
  const float width = max_pt(1) - min_pt(1) + 2 * epsilon;
  const float height = max_pt(2) - min_pt(2) + 2 * epsilon;
  origin_x_ = min_pt(0);
  origin_y_ = min_pt(1);
  origin_z_ = min_pt(2);
  resolution_ = resolution;
  occupy_threshold_ = occupy_threshold;
  raycast_step_ = resolution;

  grid_size_x_ = static_cast<int>(ceil(length / resolution_));
  grid_size_y_ = static_cast<int>(ceil(width / resolution_));
  grid_size_z_ = static_cast<int>(ceil(height / resolution_));
  grid_size_yz_ = grid_size_y_ * grid_size_z_;
  const size_t grid_total_size = static_cast<size_t>(grid_size_x_) * grid_size_y_ * grid_size_z_;
  std::vector<int> h_map(grid_total_size, 0);

  for (const auto& point : cloud->points) {
    Vector3f pt(point.x + epsilon, point.y + epsilon, point.z + epsilon);
    Vector3i vox = Pos2Vox(pt);
    if (vox.x < 0 || vox.y < 0 || vox.z < 0 || vox.x >= grid_size_x_ || vox.y >= grid_size_y_ || vox.z >= grid_size_z_) {
      continue;
    }
    h_map[Vox2Idx(vox)]++;
  }

  cudaMalloc((void**)&map_cuda_, grid_total_size * sizeof(int));
  cudaMemcpy(map_cuda_, h_map.data(), grid_total_size * sizeof(int), cudaMemcpyHostToDevice);
  return true;
}

bool GridMap::valid() const { return map_cuda_ != nullptr; }

void GridMap::freeGridMap() {
  if (map_cuda_ != nullptr) {
    cudaFree(map_cuda_);
    map_cuda_ = nullptr;
  }
}

__host__ __device__ Vector3i GridMap::Pos2Vox(const Vector3f& pos) const {
  return Vector3i(
      static_cast<int>(floor((pos.x - origin_x_) / resolution_)),
      static_cast<int>(floor((pos.y - origin_y_) / resolution_)),
      static_cast<int>(floor((pos.z - origin_z_) / resolution_)));
}

__host__ __device__ Vector3f GridMap::Vox2Pos(const Vector3i& vox) const {
  return Vector3f(
      (vox.x + 0.5f) * resolution_ + origin_x_,
      (vox.y + 0.5f) * resolution_ + origin_y_,
      (vox.z + 0.5f) * resolution_ + origin_z_);
}

__host__ __device__ int GridMap::Vox2Idx(const Vector3i& vox) const {
  return vox.x * grid_size_yz_ + vox.y * grid_size_z_ + vox.z;
}

__device__ int GridMap::mapQuery(const Vector3f& pos) const {
  Vector3i vox = Pos2Vox(pos);
  if (vox.x < 0 || vox.y < 0 || vox.z < 0 || vox.x >= grid_size_x_ || vox.y >= grid_size_y_ || vox.z >= grid_size_z_) {
    return 0;
  }
  int idx = Vox2Idx(vox);
  return map_cuda_[idx] > occupy_threshold_ ? 1 : 0;
}

GridMapView GridMap::deviceView() const {
  GridMapView view;
  view.map_cuda_ = map_cuda_;
  view.raycast_step_ = raycast_step_;
  view.resolution_ = resolution_;
  view.origin_x_ = origin_x_;
  view.origin_y_ = origin_y_;
  view.origin_z_ = origin_z_;
  view.grid_size_x_ = grid_size_x_;
  view.grid_size_y_ = grid_size_y_;
  view.grid_size_z_ = grid_size_z_;
  view.grid_size_yz_ = grid_size_yz_;
  view.occupy_threshold_ = occupy_threshold_;
  return view;
}

__global__ void cameraRaycastKernel(float* depth_values, GridMapView grid_map, CameraParams camera_param, SE3<float> T_wc) {
  int u = threadIdx.x;
  int v = blockIdx.x;
  if (u >= camera_param.image_width || v >= camera_param.image_height) {
    return;
  }

  float y = -(u - camera_param.cx) / camera_param.fx;
  float z = -(v - camera_param.cy) / camera_param.fy;
  float x = 1.0f;
  float length = sqrtf(x * x + y * y + z * z);
  x /= length; y /= length; z /= length;

  float dx = 0.5f * grid_map.raycast_step_;
  float dy = (y / x) * dx;
  float dz = (z / x) * dx;
  int scale = 0;
  float depth = camera_param.max_depth_dist;

  while (true) {
    scale += 1;
    float point_x = scale * dx;
    float point_y = scale * dy;
    float point_z = scale * dz;
    if (point_x >= camera_param.max_depth_dist) {
      depth = camera_param.max_depth_dist;
      break;
    }

    float3 point_c = make_float3(point_x, point_y, point_z);
    float3 point_w = T_wc * point_c;
    Vector3f point(point_w.x, point_w.y, point_w.z);
    if (grid_map.mapQuery(point) == 1) {
      Vector3i occ_vox_w = grid_map.Pos2Vox(point);
      Vector3f occ_point_w = grid_map.Vox2Pos(occ_vox_w);
      float3 occ_point_w3 = make_float3(occ_point_w.x, occ_point_w.y, occ_point_w.z);
      float3 occ_point_c = T_wc.inv() * occ_point_w3;
      depth = occ_point_c.x;
      break;
    }
  }

  if (depth < camera_param.min_depth_dist) {
    depth = camera_param.min_depth_dist;
  }
  depth_values[v * camera_param.image_width + u] = depth;
}

void renderDepthImage(const GridMap& grid_map, const CameraParams& camera_param, const SE3<float>& T_wc, cv::Mat& depth_image) {
  const size_t num_elements = static_cast<size_t>(camera_param.image_width) * camera_param.image_height;
  float* depth_values = nullptr;
  cudaMallocManaged(&depth_values, num_elements * sizeof(float));
  GridMapView grid_view = grid_map.deviceView();
  cameraRaycastKernel<<<camera_param.image_height, camera_param.image_width>>>(depth_values, grid_view, camera_param, T_wc);
  cudaDeviceSynchronize();

  depth_image.create(camera_param.image_height, camera_param.image_width, CV_32FC1);
  cudaMemcpy(depth_image.data, depth_values, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(depth_values);
}

}