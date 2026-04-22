cmake_minimum_required(VERSION 3.0.2)
project(external_depth_sim)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  nav_msgs
  cv_bridge
  pcl_ros
  pcl_conversions
  image_transport
  geometry_msgs
)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

set(CUDA_ARCH_FLAGS "" CACHE STRING "Optional CUDA -gencode flags, e.g. -gencode arch=compute_75,code=sm_75")
if(CUDA_ARCH_FLAGS)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH_FLAGS}")
endif()

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp sensor_msgs nav_msgs cv_bridge pcl_ros pcl_conversions image_transport geometry_msgs
  DEPENDS PCL OpenCV
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

cuda_add_library(external_depth_sim_cuda
  src/raycast_grid.cu
)
target_link_libraries(external_depth_sim_cuda
  ${CUDA_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_executable(external_depth_sim_node
  src/external_depth_sim_node.cpp
)
target_link_libraries(external_depth_sim_node
  external_depth_sim_cuda
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
)
