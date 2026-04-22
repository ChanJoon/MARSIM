#include "external_depth_sim/raycast_grid.hpp"

#include <cv_bridge/cv_bridge.h>
#include <nav_msgs/Odometry.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

#include <mutex>
#include <string>

namespace {
using external_depth_sim::CameraParams;
using external_depth_sim::GridMap;
using external_depth_sim::SE3;
using external_depth_sim::renderDepthImage;

class ExternalDepthSimNode {
 public:
  ExternalDepthSimNode() : nh_(), pnh_("~") {
    pnh_.param("drone_id", drone_id_, 0);
    pnh_.param("map_topic", map_topic_, std::string("/map_generator/global_cloud"));
    pnh_.param("odom_topic", odom_topic_, std::string("/quad_0/lidar_slam/odom"));
    pnh_.param("depth_topic", depth_topic_, defaultDepthTopic(drone_id_));
    pnh_.param("camera_info_topic", camera_info_topic_, defaultCameraInfoTopic(drone_id_));
    pnh_.param("publish_camera_info", publish_camera_info_, true);
    pnh_.param("resolution", map_resolution_, 0.1f);
    pnh_.param("occupy_threshold", occupy_threshold_, 0);
    double publish_rate;
    pnh_.param("publish_rate", publish_rate, 10.0);
    publish_period_ = ros::Duration(1.0 / publish_rate);

    pnh_.param("image_width", camera_.image_width, 160);
    pnh_.param("image_height", camera_.image_height, 90);
    pnh_.param("fx", camera_.fx, 80.0f);
    pnh_.param("fy", camera_.fy, 80.0f);
    pnh_.param("cx", camera_.cx, 80.0f);
    pnh_.param("cy", camera_.cy, 45.0f);
    pnh_.param("near_clip", camera_.min_depth_dist, 0.1f);
    pnh_.param("far_clip", camera_.max_depth_dist, 20.0f);

    map_sub_ = nh_.subscribe(map_topic_, 1, &ExternalDepthSimNode::mapCb, this);
    odom_sub_ = nh_.subscribe(odom_topic_, 1, &ExternalDepthSimNode::odomCb, this, ros::TransportHints().tcpNoDelay());
    depth_pub_ = nh_.advertise<sensor_msgs::Image>(depth_topic_, 1);
    if (publish_camera_info_) {
      camera_info_pub_ = nh_.advertise<sensor_msgs::CameraInfo>(camera_info_topic_, 1);
    }
    timer_ = nh_.createTimer(publish_period_, &ExternalDepthSimNode::timerCb, this);
  }

 private:
  static std::string defaultDepthTopic(int drone_id) {
    return "/quad" + std::to_string(drone_id) + "_external_depth_sim/depth_img";
  }
  static std::string defaultCameraInfoTopic(int drone_id) {
    return "/quad" + std::to_string(drone_id) + "_external_depth_sim/camera_info";
  }

  void mapCb(const sensor_msgs::PointCloud2ConstPtr& msg) {
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromROSMsg(*msg, cloud);
    if (cloud.empty()) {
      ROS_WARN_THROTTLE(5.0, "[external_depth_sim] received empty map cloud");
      return;
    }
    auto cloud_ptr = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>(cloud);
    GridMap new_grid;
    if (!new_grid.buildFromCloud(cloud_ptr, map_resolution_, occupy_threshold_)) {
      ROS_WARN_THROTTLE(5.0, "[external_depth_sim] failed to build grid from map cloud");
      return;
    }
    std::lock_guard<std::mutex> lock(mu_);
    grid_ = std::move(new_grid);
    map_ready_ = true;
    world_frame_id_ = msg->header.frame_id.empty() ? "world" : msg->header.frame_id;
    ROS_INFO_ONCE("[external_depth_sim] received map cloud and built occupancy grid");
  }

  void odomCb(const nav_msgs::OdometryConstPtr& msg) {
    std::lock_guard<std::mutex> lock(mu_);
    last_odom_ = msg;
  }

  void timerCb(const ros::TimerEvent&) {
    nav_msgs::OdometryConstPtr odom;
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (!map_ready_) {
        ROS_WARN_THROTTLE(5.0, "[external_depth_sim] waiting for map topic %s", map_topic_.c_str());
        return;
      }
      if (!last_odom_) {
        ROS_WARN_THROTTLE(5.0, "[external_depth_sim] waiting for odom topic %s", odom_topic_.c_str());
        return;
      }
      odom = last_odom_;
    }

    SE3<float> T_wc(
        static_cast<float>(odom->pose.pose.orientation.w),
        static_cast<float>(odom->pose.pose.orientation.x),
        static_cast<float>(odom->pose.pose.orientation.y),
        static_cast<float>(odom->pose.pose.orientation.z),
        static_cast<float>(odom->pose.pose.position.x),
        static_cast<float>(odom->pose.pose.position.y),
        static_cast<float>(odom->pose.pose.position.z));

    cv::Mat depth_image;
    renderDepthImage(grid_, camera_, T_wc, depth_image);

    cv_bridge::CvImage depth_msg;
    depth_msg.header.stamp = odom->header.stamp;
    depth_msg.header.frame_id = "/sensor";
    depth_msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
    depth_msg.image = depth_image;
    depth_pub_.publish(depth_msg.toImageMsg());

    if (publish_camera_info_) {
      sensor_msgs::CameraInfo ci;
      ci.header = depth_msg.header;
      ci.width = static_cast<uint32_t>(camera_.image_width);
      ci.height = static_cast<uint32_t>(camera_.image_height);
      ci.distortion_model = "plumb_bob";
      ci.D = {0.0, 0.0, 0.0, 0.0, 0.0};
      ci.K = {camera_.fx, 0.0, camera_.cx, 0.0, camera_.fy, camera_.cy, 0.0, 0.0, 1.0};
      ci.P = {camera_.fx, 0.0, camera_.cx, 0.0, 0.0, camera_.fy, camera_.cy, 0.0, 0.0, 0.0, 1.0, 0.0};
      camera_info_pub_.publish(ci);
    }
  }

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Subscriber map_sub_;
  ros::Subscriber odom_sub_;
  ros::Publisher depth_pub_;
  ros::Publisher camera_info_pub_;
  ros::Timer timer_;

  std::mutex mu_;
  GridMap grid_;
  nav_msgs::OdometryConstPtr last_odom_;
  bool map_ready_ = false;
  int drone_id_ = 0;
  bool publish_camera_info_ = true;
  float map_resolution_ = 0.1f;
  int occupy_threshold_ = 1;
  ros::Duration publish_period_{0.1};
  CameraParams camera_;
  std::string map_topic_;
  std::string odom_topic_;
  std::string depth_topic_;
  std::string camera_info_topic_;
  std::string world_frame_id_ = "world";
};
}  // namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "external_depth_sim_node");
  ExternalDepthSimNode node;
  ros::spin();
  return 0;
}
