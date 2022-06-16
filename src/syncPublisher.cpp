#include <syncPublisher.h>

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> syncPolicy;

SyncPublisher::SyncPublisher()
{
	// load parameters
	loadParams();
	intrinsic = cv::Mat(3, 3, CV_32FC1, cv::Scalar(0.0f));
	intrinsic.at<float>(0, 0) = 1240.8718480093278;
	intrinsic.at<float>(0, 2) = 1229.119162968315;
	intrinsic.at<float>(1, 1) = 1241.844199180136;
	intrinsic.at<float>(1, 2) = 1031.729536634529;
	intrinsic.at<float>(2, 2) = 1.0f;
	std::vector<float> distortion = {-0.0018423839142186454, 0.00023618262394699776, 0.007879783928015012, -0.0029995547312622003};

	cv::fisheye::initUndistortRectifyMap(intrinsic, distortion, cv::Mat(), intrinsic, cv::Size(2464, 2052), CV_32FC1, map1, map2);

	// publishers
	pubImg = nh.advertise<sensor_msgs::Image>("image_sync", 10);
	pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>("cloud_sync", 10);

	// ROIs = {cv::Rect(0, 0, 2464, 2052),
	// 		cv::Rect(0, 2052, 2464, 2052),
	// 		cv::Rect(0, 2052 * 2, 2464, 2052),
	// 		cv::Rect(0, 2052 * 3, 2464, 2052),
	// 		cv::Rect(0, 2052 * 4, 2464, 2052)};

	ROIs = {cv::Rect(0, 0, 2464 / 2, 2052 / 2),
			cv::Rect(0, 2052 / 2, 2464 / 2, 2052 / 2),
			cv::Rect(0, 2052 / 2 * 2, 2464 / 2, 2052 / 2),
			cv::Rect(0, 2052 / 2 * 3, 2464 / 2, 2052 / 2),
			cv::Rect(0, 2052 / 2 * 4, 2464 / 2, 2052 / 2)};
}

void SyncPublisher::loadParams()
{
	// Read image and lidar topic names from configuration file
	nh.param<std::string>("imgTopicRaw", imgTopic, "/image");
	nh.param<std::string>("laserCloudTopicRaw", laserCloudTopic, "/cloud");

	// projection params
	K << 1267.013626, 0.0, 1014.559422,
		0.0, 1267.303662, 1241.457205,
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

	cv::Rect img0ROI = ROIs[cameraNum];

	// convert from ros msg to cv img
	cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(imgMsg, sensor_msgs::image_encodings::BGR8);

	// std::cout << cv_ptr->image.size() << std::endl;
	img0 = cv_ptr->image(img0ROI);

	cv::resize(img0, img0, cv::Size(2464, 2052), 0, 0, cv::INTER_LINEAR);

	cv::remap(img0, img0, map1, map2, cv::INTER_LINEAR);
	cv::rotate(img0, img0, cv::ROTATE_90_CLOCKWISE);

	pcl::PointCloud<pcl::PointXYZ> pcdXYZ;

	// convert from ros message to pcl point cloud
	pcl::fromROSMsg(*laserCloudMsg, pcdXYZ);
	pcl::PointCloud<pcl::PointXYZRGB> pcdXYZRGB;

	// std::cout << "The number of pts: " << pcdXYZ.size() << std::endl;

	Eigen::MatrixXf P = K * Tcl; // projection matrix

	//#pragma omp parallel for num_threads(10)
	for (int i = 0; i < pcdXYZ.size(); i++)
	{

		pcl::PointXYZ p = pcdXYZ[i];

		float theta = 180.0 / M_PI * atan2(p.y, p.x);

		int isValidPoint = ((cameraNum == 0 && abs(theta) < 36.0 - angleGap) ||
							(cameraNum == 1 && (theta > -108.0 + angleGap && theta < -36.0 - angleGap)) ||
							(cameraNum == 2 && (theta > -180.0 + angleGap && theta < -108.0 - angleGap)) ||
							(cameraNum == 3 && (theta > 108.0 + angleGap && theta < 180.0 - angleGap)) ||
							(cameraNum == 4 && (theta > 36.0 + angleGap && theta < 108.0 - angleGap)));

		double dist = sqrt(pow(p.x, 2) + pow(p.y, 2) + pow(p.z, 2));
		if (isValidPoint && dist < maxRange && p.z > minZ && p.z < maxZ)
		{

			pcl::PointXYZRGB pc;

			pc.x = p.x;
			pc.y = p.y;
			pc.z = p.z;
			// pc.r = 255.0;
			// pc.g = 0.0;
			// pc.b = 0.0;

			pcdXYZRGB.push_back(pc);

			Eigen::Vector4f phat(pc.x, pc.y, pc.z, 1.0);
			Eigen::Vector3f xys = P * phat;
			xys /= xys(2);

			if (xys[0] > 0 && xys[0] < 2051 && xys[1] > 0 && xys[1] < 2453)
			{
				cv::Vec3b rgb = img0.at<cv::Vec3b>(xys[1], xys[0]);

				pc.r = (double)rgb[0];
				pc.g = (double)rgb[1];
				pc.b = (double)rgb[2];

				pcdXYZRGB.push_back(pc);
			}
		}
	}

	// std::cout << "TIME 0~1: " << (double)(t1 - t0) / CLOCKS_PER_SEC << std::endl;
	// std::cout << "TIME 1~2: " << (double)(t2 - t1) / CLOCKS_PER_SEC << std::endl;
	// std::cout << "TIME 2~3: " << (double)(t3 - t2) / CLOCKS_PER_SEC << std::endl;
	// std::cout << "TIME pcd: " << (double)(end - start) / CLOCKS_PER_SEC << std::endl;

	// distortion rectification & rotate

	// convert from cv img to ros msg
	// sensor_msgs::Image::Ptr thisImgRectifiedPtr = (cv_ptr->toImageMsg());
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
