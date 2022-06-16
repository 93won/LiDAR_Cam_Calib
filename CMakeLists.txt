cmake_minimum_required(VERSION 3.8.3)
project(lidar_cam_calib)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -pthread")


find_package(catkin REQUIRED COMPONENTS
  tf
  roscpp
  rospy
  cv_bridge
  # pcl library
  pcl_ros
  pcl_conversions
  # msgs
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  message_generation
  visualization_msgs
  message_filters
)

find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters)
find_package(Ceres REQUIRED)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Ceres_INCLUDE_DIRS}
)

link_directories(
	include
	${PCL_LIBRARY_DIRS}
  ${OpenCV_LIBRARY_DIRS}
  ${Eigen3_LIBRARY_DIRS}
  ${GTSAM_LIBRARY_DIRS}
  ${CERES_LIBRARIES}
)
generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
  nav_msgs
  sensor_msgs
  visualization_msgs
)

catkin_package(
  INCLUDE_DIRS include
  DEPENDS PCL

  CATKIN_DEPENDS
  std_msgs
  nav_msgs
  geometry_msgs
  sensor_msgs
  visualization_msgs
  message_runtime
  message_generation
  visualization_msgs
)



add_executable(lidarCameraCalib src/lidarCameraCalib.cpp)
target_link_libraries(lidarCameraCalib ${catkin_LIBRARIES} ${Eigen3_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARY_DIRS} ${CERES_LIBRARIES} ${OpenMP_CXX_FLAGS})
add_dependencies(lidarCameraCalib ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(lidarCameraCalibMono src/lidarCameraCalibMono.cpp)
target_link_libraries(lidarCameraCalibMono ${catkin_LIBRARIES} ${Eigen3_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARY_DIRS} ${CERES_LIBRARIES} ${OpenMP_CXX_FLAGS})
add_dependencies(lidarCameraCalibMono ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})


add_executable(syncPublisher src/syncPublisher.cpp)
target_link_libraries(syncPublisher ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARY_DIRS} ${OpenMP_CXX_FLAGS})
add_dependencies(syncPublisher ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(syncPublisherMono src/syncPublisherMono.cpp)
target_link_libraries(syncPublisherMono ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARY_DIRS} ${OpenMP_CXX_FLAGS})
add_dependencies(syncPublisherMono ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

