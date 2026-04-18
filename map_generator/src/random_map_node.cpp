// random_map_node.cpp
//
// Seeded random-map generator for MARSIM. Publishes a latched
// sensor_msgs/PointCloud2 on /map_generator/global_cloud (same topic as
// map_pub) and optionally saves a scratch PCD so the GPU renderer
// (opengl_render_node) can load the same scene from disk.
//
// Scene backends:
//   scene_type=0 : built-in cylinder forest (minimal, no extra deps).
//   scene_type=1 : perlin3D cave        (via mocka::Maps, ported from YOPO)
//   scene_type=2 : random column forest (YOPO-style, with ground)
//   scene_type=3 : maze2D
//   scene_type=4 : Maze3D
//   scene_type=5 : tree forest (loads yopo_tree.ply)
//   scene_type=6 : room
//   scene_type=7 : wall
//
// ROS private params common to all scenes:
//   ~seed              int,    default 0  (0 => wall-clock, non-reproducible)
//   ~map_size_x/y/z    double, default 40/40/3  (meters)
//   ~resolution        double, default 0.1
//   ~pcd_out_path      string, default ""  (if non-empty, save PCD there)
//   ~scene_type        int,    default 0
//   ~use_ground_plane  bool,   default false  (Branch 1: inject z=ground_z grid)
//   ~ground_resolution double, default 0.1    (ground grid spacing, meters)
//   ~ground_z          double, default 0.0    (ground plane altitude, meters)
//
// scene_type=0 only:
//   ~obstacle_num / ~obstacle_radius_min/max / ~obstacle_height_min/max /
//   ~keepout_radius
//
// scene_type=1..7: see yopo_maps/maps.cpp for the underlying params. We
// expose each one as a ROS param with the upstream default:
//   ~complexity ~fill ~fractal ~attenuation
//   ~width_min ~width_max ~obstacle_number
//   ~road_width ~add_wall_x ~add_wall_y
//   ~tree_file ~tree_dist
//   ~room_number ~max_windows ~add_ceiling ~window_size_min ~window_size_max
//   ~wall_width_min ~wall_width_max ~wall_thick ~wall_number ~wall_ceiling

#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <yaml-cpp/yaml.h>

#include <random>
#include <string>
#include <cmath>

#include "yopo_maps/maps.hpp"
#include "ground_plane.hpp"

namespace {

// --- Built-in cylinder forest (scene_type=0) -------------------------------
void generateCylinderForest(pcl::PointCloud<pcl::PointXYZ>& cloud,
                            uint32_t actual_seed,
                            double map_size_x, double map_size_y, double map_size_z,
                            int obstacle_num,
                            double r_min, double r_max,
                            double h_min, double h_max,
                            double resolution,
                            double keepout_radius) {
  std::mt19937 rng(actual_seed);
  std::uniform_real_distribution<double> xdist(-map_size_x * 0.5, map_size_x * 0.5);
  std::uniform_real_distribution<double> ydist(-map_size_y * 0.5, map_size_y * 0.5);
  std::uniform_real_distribution<double> rdist(r_min, r_max);
  std::uniform_real_distribution<double> hdist(h_min, h_max);

  cloud.points.reserve(obstacle_num * 200);
  int placed   = 0;
  int attempts = 0;
  const int max_attempts = obstacle_num * 20;
  while (placed < obstacle_num && attempts < max_attempts) {
    ++attempts;
    const double cx = xdist(rng);
    const double cy = ydist(rng);
    if (std::hypot(cx, cy) < keepout_radius) continue;

    const double radius = rdist(rng);
    const double height = std::min(hdist(rng), map_size_z);
    const int nr = std::max(1, static_cast<int>(std::ceil(radius / resolution)));
    const int nz = std::max(1, static_cast<int>(std::ceil(height / resolution)));
    for (int iz = 0; iz < nz; ++iz) {
      const double z = iz * resolution;
      for (int ix = -nr; ix <= nr; ++ix) {
        for (int iy = -nr; iy <= nr; ++iy) {
          const double dx = ix * resolution;
          const double dy = iy * resolution;
          if (dx * dx + dy * dy > radius * radius) continue;
          cloud.points.emplace_back(pcl::PointXYZ(
              static_cast<float>(cx + dx),
              static_cast<float>(cy + dy),
              static_cast<float>(z)));
        }
      }
    }
    ++placed;
  }
  if (placed < obstacle_num) {
    ROS_WARN("[random_map_node] only placed %d / %d cylinders after %d attempts",
             placed, obstacle_num, attempts);
  }
}

// --- YOPO mocka::Maps dispatch (scene_type=1..7) ---------------------------
void generateYopoScene(pcl::PointCloud<pcl::PointXYZ>& cloud,
                       ros::NodeHandle& nh,
                       int scene_type,
                       uint32_t actual_seed,
                       double map_size_x, double map_size_y, double map_size_z,
                       double resolution) {
  // Build a YAML::Node in-memory from ROS params so we can reuse
  // mocka::Maps::setParam verbatim.
  YAML::Node cfg;
  // perlin3D
  cfg["complexity"]   = nh.param<double>("complexity", 0.02);
  cfg["fill"]         = nh.param<double>("fill", 0.1);
  cfg["fractal"]      = nh.param<int>("fractal", 1);
  cfg["attenuation"]  = nh.param<double>("attenuation", 0.1);
  // random columns
  cfg["width_min"]        = nh.param<double>("width_min", 0.6);
  cfg["width_max"]        = nh.param<double>("width_max", 1.5);
  cfg["obstacle_number"]  = nh.param<int>("obstacle_number", 100);
  // maze2D
  cfg["road_width"]   = nh.param<double>("road_width", 3.0);
  cfg["add_wall_x"]   = nh.param<int>("add_wall_x", 1);
  cfg["add_wall_y"]   = nh.param<int>("add_wall_y", 1);
  // forest
  const std::string default_tree =
      ros::package::getPath("map_generator") + "/resource/yopo_tree.ply";
  cfg["tree_file"]    = nh.param<std::string>("tree_file", default_tree);
  cfg["tree_dist"]    = nh.param<double>("tree_dist", 4.0);
  // room
  cfg["room_number"]      = nh.param<int>("room_number", 4);
  cfg["max_windows"]      = nh.param<int>("max_windows", 2);
  cfg["add_ceiling"]      = nh.param<int>("add_ceiling", 0);
  cfg["window_size_min"]  = nh.param<double>("window_size_min", 2.0);
  cfg["window_size_max"]  = nh.param<double>("window_size_max", 2.8);
  // wall
  cfg["wall_width_min"]   = nh.param<double>("wall_width_min", 0.5);
  cfg["wall_width_max"]   = nh.param<double>("wall_width_max", 6.0);
  cfg["wall_thick"]       = nh.param<double>("wall_thick", 0.5);
  cfg["wall_number"]      = nh.param<int>("wall_number", 100);
  cfg["wall_ceiling"]     = nh.param<int>("wall_ceiling", 1);

  mocka::Maps::BasicInfo info;
  // mocka grid sizes are in cells at given scale (scale = points/meter).
  info.scale = 1.0 / resolution;
  info.sizeX = static_cast<int>(std::round(map_size_x * info.scale));
  info.sizeY = static_cast<int>(std::round(map_size_y * info.scale));
  info.sizeZ = static_cast<int>(std::round(map_size_z * info.scale));
  info.seed  = static_cast<int>(actual_seed);
  info.cloud = cloud.makeShared();  // takes a shared_ptr alias; we'll swap below

  // Actually use a fresh shared_ptr the generator can push into.
  pcl::PointCloud<pcl::PointXYZ>::Ptr out(new pcl::PointCloud<pcl::PointXYZ>());
  info.cloud = out;

  mocka::Maps gen;
  gen.setParam(cfg);
  gen.setInfo(info);
  gen.generate(scene_type);

  cloud.swap(*out);
  cloud.width  = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
}

}  // namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "random_map_node");
  ros::NodeHandle nh("~");

  int    seed           = 0;
  int    scene_type     = 0;
  double map_size_x     = 40.0;
  double map_size_y     = 40.0;
  double map_size_z     = 3.0;
  int    obstacle_num   = 120;
  double r_min          = 0.2;
  double r_max          = 0.5;
  double h_min          = 1.0;
  double h_max          = 3.0;
  double resolution     = 0.1;
  double keepout_radius = 2.0;
  bool   use_ground_plane  = false;
  double ground_resolution = 0.1;
  double ground_z          = 0.0;
  std::string pcd_out_path;

  nh.param("seed",                 seed,           seed);
  nh.param("scene_type",           scene_type,     scene_type);
  nh.param("map_size_x",           map_size_x,     map_size_x);
  nh.param("map_size_y",           map_size_y,     map_size_y);
  nh.param("map_size_z",           map_size_z,     map_size_z);
  nh.param("obstacle_num",         obstacle_num,   obstacle_num);
  nh.param("obstacle_radius_min",  r_min,          r_min);
  nh.param("obstacle_radius_max",  r_max,          r_max);
  nh.param("obstacle_height_min",  h_min,          h_min);
  nh.param("obstacle_height_max",  h_max,          h_max);
  nh.param("resolution",           resolution,     resolution);
  nh.param("keepout_radius",       keepout_radius, keepout_radius);
  nh.param("use_ground_plane",     use_ground_plane,  use_ground_plane);
  nh.param("ground_resolution",    ground_resolution, ground_resolution);
  nh.param("ground_z",             ground_z,          ground_z);
  nh.param<std::string>("pcd_out_path", pcd_out_path, std::string(""));

  uint32_t actual_seed;
  if (seed == 0) {
    actual_seed = static_cast<uint32_t>(ros::Time::now().toNSec() & 0xffffffffu);
    ROS_WARN("[random_map_node] seed=0 -> wall-clock seed=%u (NON-REPRODUCIBLE)", actual_seed);
  } else {
    actual_seed = static_cast<uint32_t>(seed);
    ROS_INFO("[random_map_node] deterministic seed=%u scene_type=%d", actual_seed, scene_type);
  }

  pcl::PointCloud<pcl::PointXYZ> cloud;

  if (scene_type == 0) {
    generateCylinderForest(cloud, actual_seed,
                           map_size_x, map_size_y, map_size_z,
                           obstacle_num, r_min, r_max, h_min, h_max,
                           resolution, keepout_radius);
  } else if (scene_type >= 1 && scene_type <= 7) {
    generateYopoScene(cloud, nh, scene_type, actual_seed,
                      map_size_x, map_size_y, map_size_z, resolution);
  } else {
    ROS_ERROR("[random_map_node] unsupported scene_type=%d (expected 0..7)", scene_type);
    return 1;
  }

  if (use_ground_plane) {
    const size_t before = cloud.points.size();
    map_generator::appendGroundPlane(cloud, map_size_x, map_size_y,
                                     ground_resolution, ground_z);
    ROS_INFO("[random_map_node] ground plane: +%zu points at z=%.3f (res=%.3f)",
             cloud.points.size() - before, ground_z, ground_resolution);
  }

  // Voxel downsample to target resolution.
  {
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(cloud.makeShared());
    vg.setLeafSize(static_cast<float>(resolution),
                   static_cast<float>(resolution),
                   static_cast<float>(resolution));
    pcl::PointCloud<pcl::PointXYZ> tmp;
    vg.filter(tmp);
    cloud.swap(tmp);
  }
  cloud.width    = cloud.points.size();
  cloud.height   = 1;
  cloud.is_dense = true;
  ROS_INFO("[random_map_node] generated %zu points (scene_type=%d, seed=%u)",
           cloud.points.size(), scene_type, actual_seed);

  if (!pcd_out_path.empty()) {
    if (pcl::io::savePCDFileBinary(pcd_out_path, cloud) == 0) {
      ROS_INFO("[random_map_node] wrote PCD: %s", pcd_out_path.c_str());
    } else {
      ROS_ERROR("[random_map_node] failed to write PCD: %s", pcd_out_path.c_str());
    }
  }

  ros::Publisher cloud_pub = nh.advertise<sensor_msgs::PointCloud2>(
      "/map_generator/global_cloud", 1, /*latch=*/true);
  sensor_msgs::PointCloud2 msg;
  pcl::toROSMsg(cloud, msg);
  msg.header.frame_id = "world";
  msg.header.stamp    = ros::Time::now();

  ros::Rate r(1.0);
  int ticks = 0;
  while (ros::ok()) {
    cloud_pub.publish(msg);
    ros::spinOnce();
    r.sleep();
    if (++ticks > 10) break;
  }

  ros::spin();
  return 0;
}
