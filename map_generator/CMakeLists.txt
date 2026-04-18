cmake_minimum_required(VERSION 2.8.3)
project(map_generator)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  roslib
  rospy
  std_msgs
  geometry_msgs
  pcl_conversions
)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp std_msgs
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
  src
)

add_executable (map_pub src/map_publisher.cpp )
target_link_libraries(map_pub
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES})

add_executable (random_map_node
  src/random_map_node.cpp
  src/yopo_maps/maps.cpp
  src/yopo_maps/perlinnoise.cpp)
target_link_libraries(random_map_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
  yaml-cpp)
