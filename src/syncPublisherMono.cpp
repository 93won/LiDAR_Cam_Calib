#include <syncPublisher.h>

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> syncPolicy;

SyncPublisher::SyncPublisher()
{
	// load parameters
	loadParams();
	intrinsic = cv::Mat(3, 3, CV_32FC1, cv::Scalar(0.0f));
	intrinsic.at<float>(0, 0) = 1064.178810;
	intrinsic.at<float>(0, 2) = 653.523533;
	intrinsic.at<float>(1, 1) = 1063.842836;
	intrinsic.at<float>(1, 2) = 341.754623;
	intrinsic.at<float>(2, 2) = 1.0f;
	std::vector<float> distortion = {0.049428, 1.337325, -0.000269, 0.007203};

	cv::fisheye::initUndistortRectifyMap(intrinsic, distortion, cv::Mat(), intrinsic, cv::Size(1280, 720), CV_32FC1, map1, map2);

	// publishers
	pubImg = nh.advertise<sensor_msgs::Image>("image_sync", 10);
	pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>("cloud_sync", 10);
}

void SyncPublisher::loadParams()
{
	// Read image and lidar topic names from configuration file
	nh.param<std::string>("imgTopicRaw", imgTopic, "/image");
	nh.param<std::string>("laserCloudTopicRaw", laserCloudTopic, "/cloud");

	// projection params
	K << 1064.178810, 0.0, 653.523533,
		0.0, 1063.842836, 341.754623,
		0.0, 0.0, 1.0;

	std::vector<double> extRotLC0, extTransLC0;
	nh.param<std::vector<double>>("extrinsicRotLC0", extRotLC0, std::vector<double>());
	nh.param<std::vector<double>>("extrinsicTransLC0", extTransLC0, std::vector<double>());

	Tcl = Eigen::MatrixXf(3, 4);
	Tcl << extRotLC0[0], extRotLC0[1], extRotLC0[2], extTransLC0[0],
		extRotLC0[3], extRotLC0[4], extRotLC0[5], extTransLC0[1],
		extRotLC0[6], extRotLC0[7], extRotLC0[8], extTransLC0[2];

	// camera number
	nh.param<int>("cameraNum", cameraNum, 0);

	nh.param<double>("maxRange", maxRange, 1.0);
	nh.param<double>("minZ", minZ, -0.3);
	nh.param<double>("maxZ", maxZ, 0.3);

	nh.param<double>("maxDist", maxDist, 0.05);
	nh.param<int>("minInliers", minInliers, 100);
	nh.param<double>("squareRange", squareRange, 0.7);

	nh.param<double>("eplisonZ", eplisonZ, 0.1);
	nh.param<double>("maxDistLine", maxDistLine, 0.01);
	nh.param<int>("numFrag", numFrag, 100);
	nh.param<int>("numInliersLine", numInliersLine, 5);

	nh.param<double>("angleGap", angleGap, 10.0);
}

void SyncPublisher::imgCloudSyncCallback(const sensor_msgs::ImageConstPtr &imgMsg, const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{
	// std::cout << "I'm SyncPublisher!" << std::endl;
	// std::cout << "LIDAR TIME: " << std::setprecision(18) << static_cast<double>(laserCloudMsg->header.stamp.toSec()) << std::endl;
	// std::cout << "IMAGE TIME: " << std::setprecision(18) << static_cast<double>(imgMsg->header.stamp.toSec()) << std::endl;

	sensor_msgs::Image thisImg = *imgMsg;
	cv::Mat img0;

	// convert from ros msg to cv img
	cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(imgMsg, sensor_msgs::image_encodings::BGR8);

	// std::cout << cv_ptr->image.size() << std::endl;
	img0 = cv_ptr->image;

	// cv::remap(img0, img0, map1, map2, cv::INTER_LINEAR);

	pcl::PointCloud<pcl::PointXYZ> pcdXYZ;

	// convert from ros message to pcl point cloud
	pcl::fromROSMsg(*laserCloudMsg, pcdXYZ);
	pcl::PointCloud<pcl::PointXYZRGB> pcdXYZRGB;

	Eigen::MatrixXf P = K * Tcl; // projection matrix

	//#pragma omp parallel for num_threads(10)
	for (int i = 0; i < pcdXYZ.size(); i++)
	{

		pcl::PointXYZ p = pcdXYZ[i];

		float theta = 180.0 / M_PI * atan2(p.y, p.x);

		bool isValidPoint = (abs(theta) < 90);

		double dist = sqrt(pow(p.x, 2) + pow(p.y, 2) + pow(p.z, 2));

		if (isValidPoint && dist < maxRange && p.z > minZ && p.z < maxZ)
		{

			pcl::PointXYZRGB pc;

			pc.x = p.x;
			pc.y = p.y;
			pc.z = p.z;

			// pcdXYZRGB.push_back(pc);

			// colorize

			Eigen::Vector4f phat(pc.x, pc.y, pc.z, 1.0);
			Eigen::Vector3f xys = P * phat;
			xys /= xys(2);

			if (xys[0] > 0 && xys[0] < 1279 && xys[1] > 0 && xys[1] < 719)
			{
				cv::Vec3b rgb = img0.at<cv::Vec3b>(xys[1], xys[0]);

				// bgr
				pc.b = (double)rgb[0];
				pc.g = (double)rgb[1];
				pc.r = (double)rgb[2];

				// std::cout << "CHECK RGB: " << pc.r << " " << pc.g << " " << pc.b << std::endl;

				if (pc.r < 10.0 && pc.g < 10.0 && pc.b < 10.0)
				{
				}
				else
				{
					pcdXYZRGB.push_back(pc);
				}
			}
			else
			{
				// std::cout << "Not in range! // " << xys[0] << " " << xys[1] << std::endl;
			}
		}
	}

	sensor_msgs::PointCloud2 thisLaserCloud;
	pcl::toROSMsg(pcdXYZRGB, thisLaserCloud);
	thisLaserCloud.header.frame_id = "/lidar";
	thisLaserCloud.header.stamp = laserCloudMsg->header.stamp;

	// publish
	pubLaserCloud.publish(thisLaserCloud);

	cv_bridge::CvImage img_bridge;
	sensor_msgs::Image img_msg; // >> message to be sent

	std_msgs::Header header; // empty header
	header.stamp = imgMsg->header.stamp;
	img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, img0);
	img_bridge.toImageMsg(img_msg); // from cv_bridge to sensor_msgs::Image
	pubImg.publish(img_msg);		// ros::Publisher pub_img = node.advertise<sensor_msgs::Image>("topic", queuesize);
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "sync_publisher");
	ros::NodeHandle nh_;

	SyncPublisher syncPub;

	message_filters::Subscriber<sensor_msgs::Image> imgSubSync(nh_, syncPub.imgTopic, 40);
	message_filters::Subscriber<sensor_msgs::PointCloud2> laserCloudSubSync(nh_, syncPub.laserCloudTopic, 40);

	message_filters::Synchronizer<syncPolicy> sync(syncPolicy(10), imgSubSync, laserCloudSubSync);
	sync.registerCallback(boost::bind(&SyncPublisher::imgCloudSyncCallback, &syncPub, _1, _2));

	ros::spin();
	return 0;
}
