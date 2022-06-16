#ifndef LIDARCAMCALIB_H
#define LIDARCAMCALIB_H

// ROS
#include <sensor_msgs/PointCloud2.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <visualization_msgs/MarkerArray.h>

// Eigen
#include <eigen3/Eigen/Dense>

// OpenCV
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>

// STL
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>

struct EIGEN_ALIGN16 OusterPointXYZIRT
{
	PCL_ADD_POINT4D
	PCL_ADD_INTENSITY
	std::uint16_t ring; // use std::uint for namespace collision with pcl
	double time;
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
POINT_CLOUD_REGISTER_POINT_STRUCT(OusterPointXYZIRT,
								  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(std::uint16_t, ring, ring)(double, time, time))

struct EIGEN_ALIGN16 OusterPointXYZRGBIRT
{
	PCL_ADD_POINT4D // This adds the members x,y,z which can also be accessed using the point (which is float[4])
	PCL_ADD_RGB
	PCL_ADD_INTENSITY
	std::uint16_t ring; // use std::uint for namespace collision with pcl
	double time;
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
POINT_CLOUD_REGISTER_POINT_STRUCT(OusterPointXYZRGBIRT,
								  (float, x, x)(float, y, y)(float, z, z)(float, rgb, rgb)(float, intensity, intensity)(std::uint16_t, ring, ring)(double, time, time))

using PointXYZIRT = OusterPointXYZIRT;
using PointXYZRGBIRT = OusterPointXYZRGBIRT;

class LidarCamCalib
{
private:
	// ROS Standard Variables
	ros::NodeHandle nh;

	// ROS Publisher and Subscriber
	ros::Subscriber subLaserCloud, subImg;
	ros::Publisher pubLaserCloud, pubImg;

	ros::Publisher pubTransformedCloud;
	ros::Publisher pubLineCloud;
	ros::Publisher pubLineEdge;
	ros::Publisher pubCheckerEdge;
	ros::Publisher pubMarkerArray;

	ros::Publisher pubPlaneAll;

	ros::Publisher pubDebug;

	std::mutex imgLock;
	std::mutex laserCloudLock;

	std::deque<sensor_msgs::Image> imgQueue;
	std::deque<sensor_msgs::PointCloud2> laserCloudQueue;

public:
	LidarCamCalib();
	~LidarCamCalib(){};

	// Parameter Server
	void loadParams();

	// General Methods
	void run();

	// Topic name
	std::string imgTopic;
	std::string laserCloudTopic;

	// Point cloud filter params
	double maxRange, minZ, maxZ;

	// Plane segmentation params
	double maxDist, squareRange;
	int minInliers;

	// Line detectin params
	double eplisonZ;
	double maxDistLine;
	int numFrag;
	int numInliersLine;

	// check
	int numImg;

	// Callback funstions
	void sensorDataHandler(const sensor_msgs::ImageConstPtr &imgMsg, const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg);
	void imgCloudSyncCallback(const sensor_msgs::ImageConstPtr &imgMsg, const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg);

	// Intrinsics & Distortion rectification maps
	cv::Mat intrinsic; //(3, 3, CV_32FC1, cv::Scalar(0.0f));
	cv::Mat map1, map2;

	Eigen::Matrix3f K;
	Eigen::MatrixXf Tcl;

	// Optimization targets
	std::vector<std::vector<cv::Point2f>> cornersTotal;
	std::vector<std::vector<Eigen::Vector3f>> pointsCheckerTotal;

	int cameraNum;
};

class CvImage
{
public:
	std_msgs::Header header;
	std::string encoding;
	cv::Mat image;
};

typedef boost::shared_ptr<CvImage const> CvImageConstPtr;

#endif
