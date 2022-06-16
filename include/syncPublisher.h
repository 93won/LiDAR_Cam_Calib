#ifndef SYNCPUBLISHER_H
#define SYNCPUBLISHER_H


// ROS
#include <sensor_msgs/PointCloud2.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

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




class SyncPublisher
{
private:
	// ROS Standard Variables
	ros::NodeHandle nh;

	// ROS publisher
	ros::Publisher pubLaserCloud, pubImg;

public:
	SyncPublisher();
	~SyncPublisher(){};

	// Parameter Server
	void loadParams();

	// Topic name
	std::string imgTopic;
	std::string laserCloudTopic;

	void imgCloudSyncCallback(const sensor_msgs::ImageConstPtr &imgMsg, const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg);

	// Intrinsics & Distortion rectification maps
	cv::Mat intrinsic; //(3, 3, CV_32FC1, cv::Scalar(0.0f));
	cv::Mat map1, map2;

	// Intrinsic(rectified) & Extrinsic (lidar->cam0)
	Eigen::Matrix3f K;
	Eigen::MatrixXf Tcl;

	// Camera number
	int cameraNum;

	// ROI per each camera
	std::vector<cv::Rect> ROIs;

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

	// angle filtering...
	double angleGap;

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
