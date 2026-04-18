// ground_plane.hpp
//
// Conditional ground-plane point injection for MARSIM scenes.
//
// MARSIM renders point clouds via GL_POINTS (GPU `opengl_render_node`) or
// ray-casts against them (CPU `pcl_render_node`). There is no mesh / textured
// quad path. "Ground plane" in this fork therefore means injecting a dense
// z=const grid of points into the published `global_map` cloud so both
// renderers see it and planners avoid it.
//
// Default `use_ground_plane=false` must be bit-for-bit the legacy behavior —
// all helpers here are no-ops unless the caller explicitly opts in.

#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <cmath>

namespace map_generator {

// Append a dense z=ground_z grid across [-size_x/2, +size_x/2] x [-size_y/2, +size_y/2]
// with spacing `resolution`. Extents and resolution are in meters. Callers
// should guard this call behind a `use_ground_plane` flag so the legacy
// (no-ground) path is bit-for-bit preserved.
inline void appendGroundPlane(pcl::PointCloud<pcl::PointXYZ>& cloud,
                              double size_x,
                              double size_y,
                              double resolution,
                              double ground_z) {
  if (resolution <= 0.0 || size_x <= 0.0 || size_y <= 0.0) return;

  const double half_x = 0.5 * size_x;
  const double half_y = 0.5 * size_y;
  const int nx = static_cast<int>(std::floor(size_x / resolution)) + 1;
  const int ny = static_cast<int>(std::floor(size_y / resolution)) + 1;

  cloud.points.reserve(cloud.points.size() +
                       static_cast<size_t>(nx) * static_cast<size_t>(ny));

  for (int iy = 0; iy < ny; ++iy) {
    const double y = -half_y + iy * resolution;
    for (int ix = 0; ix < nx; ++ix) {
      const double x = -half_x + ix * resolution;
      cloud.points.emplace_back(pcl::PointXYZ(static_cast<float>(x),
                                              static_cast<float>(y),
                                              static_cast<float>(ground_z)));
    }
  }
  cloud.width    = cloud.points.size();
  cloud.height   = 1;
  cloud.is_dense = true;
}

}  // namespace map_generator
