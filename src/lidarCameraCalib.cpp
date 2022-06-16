#include <iostream>
#include <algorithm>
#include <lidarCameraCalib.h>
#include <ceres/ceres.h>
#include <factor.h>
#include <pcl/filters/statistical_outlier_removal.h>

LidarCamCalib::LidarCamCalib()
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
	pubImg = nh.advertise<sensor_msgs::Image>("image_debug", 10);
	pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>("cloud_debug", 10);
	pubDebug = nh.advertise<sensor_msgs::PointCloud2>("segmentation_target", 10);

	pubTransformedCloud = nh.advertise<sensor_msgs::PointCloud2>("transform_debug", 10);
	pubLineCloud = nh.advertise<sensor_msgs::PointCloud2>("line_target_cloud", 10);
	pubLineEdge = nh.advertise<sensor_msgs::PointCloud2>("line_edge_debug", 10);
	pubMarkerArray = nh.advertise<visualization_msgs::MarkerArray>("marker_debug", 1);
	pubCheckerEdge = nh.advertise<sensor_msgs::PointCloud2>("checker_edge_debug", 10);
	pubPlaneAll = nh.advertise<sensor_msgs::PointCloud2>("all_plane_points", 10);

	// check
	numImg = 0;

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
}

void LidarCamCalib::loadParams()
{
	// Read image and lidar topic names from configuration file
	nh.param<std::string>("imgTopicSync", imgTopic, "/image");
	nh.param<std::string>("laserCloudTopicSync", laserCloudTopic, "/cloud");

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

	nh.param<int>("cameraNum", cameraNum, 0);
}

void LidarCamCalib::sensorDataHandler(const sensor_msgs::ImageConstPtr &imgMsg, const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{
	std::vector<cv::Point2f> corners;
	std::vector<Eigen::Vector3f> pointsChecker;

	sensor_msgs::Image thisImg = *imgMsg;

	// convert from ros msg to cv img
	cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(imgMsg, sensor_msgs::image_encodings::BGR8);

	if (numImg % 10 == 0)
	{
		std::string imgPath = "/home/maxst/Data/LiDAR_Cam_Calib/" + std::to_string(numImg) + ".jpg";
		cv::imwrite(imgPath, cv_ptr->image);
	}
	numImg += 1;

	// detect checker board
	cv::Mat gray;
	cv::Size patternSize(10, 8);

	cv::cvtColor(cv_ptr->image, gray, cv::COLOR_BGR2GRAY);

	bool patternFound = cv::findChessboardCorners(gray, patternSize, corners, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);

	pcl::PointCloud<PointXYZIRT> pcdXYZIRT;
	pcl::PointCloud<PointXYZRGBIRT> pcdXYZRGBIRT;
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcdXYZ(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcdXYZOrig(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcdXYZFilterd(new pcl::PointCloud<pcl::PointXYZ>);

	// for debug
	pcl::PointCloud<pcl::PointXYZ> pcdLine2;
	pcl::PointCloud<pcl::PointXYZ> pcdEdge;
	pcl::PointCloud<pcl::PointXYZRGB> pcdEdgeColor;
	visualization_msgs::MarkerArray checkerLine;
	pcl::PointCloud<pcl::PointXYZRGB> pcdCheckerEdgeColor;

	// convert from ros message to pcl point cloud
	pcl::fromROSMsg(*laserCloudMsg, *pcdXYZ);
	pcl::fromROSMsg(*laserCloudMsg, *pcdXYZOrig);

	// filtering
	for (auto &p : *pcdXYZ)
	{
		pcl::PointXYZ pp;
		pp.x = p.x;
		pp.y = p.y;
		pp.z = p.z;

		double dist = sqrt(pow(pp.x, 2.0) + pow(pp.y, 2.0) + pow(pp.z, 2.0));

		if (dist < maxRange && p.z > minZ && p.z < maxZ)
			pcdXYZFilterd->push_back(pp);
	}

	pcdXYZ = pcdXYZFilterd;

	// checker board plane variables
	pcl::PointCloud<pcl::PointXYZRGB> pcdXYZRGB;	   // for debug segmented plane
	pcl::PointCloud<pcl::PointXYZRGB> pcdXYZRGB2;	   // for debug segmented plane
	pcl::PointCloud<pcl::PointXYZRGB> pcdXYZRGBCenter; // for debug segmented plane
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcdXYZCheckerboard(new pcl::PointCloud<pcl::PointXYZ>);
	Eigen::MatrixXf planeParams(4, 1);
	// std::vector<double> planeParams(4);

	std::vector<std::vector<double>> colors = {{255.0, 0.0, 0.0}, {0.0, 255.0, 0.0}, {0.0, 0.0, 255.0}};
	int nbPlanes = 0;
	int maxIter = 30;

	// plane fitting
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
	pcl::SACSegmentation<pcl::PointXYZ> seg;
	pcl::ExtractIndices<pcl::PointXYZ> extract;
	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setDistanceThreshold(maxDist);

	int iter = 0;

	float cx, cy, cz;

	std::vector<float> coeff;

	pcl::PointCloud<pcl::PointXYZ> pcdPlane, pcdSegTarget; // for debug segmented plane

	pcdSegTarget = *pcdXYZ;

	std::cout << "The number of segmentation target: " << pcdSegTarget.size() << " // Orig: " << (*pcdXYZOrig).size() << std::endl;

	std::cout << "##########################################################" << std::endl;
	while (true)
	{
		if (iter > maxIter)
			break;

		// segmentation
		seg.setInputCloud(pcdXYZ);
		seg.segment(*inliers, *coefficients);

		// check result
		coeff = coefficients->values;

		// assume checkerboard plane is vertical plane

		if (inliers->indices.size() >= minInliers && coeff[2] < 0.5)
		{

			for (int i = 0; i < inliers->indices.size(); i++)
			{
				if (i == 0)
				{
					cx = 0;
					cy = 0;
					cz = 0;
				}
				pcl::PointXYZ p = pcdXYZ->points[inliers->indices[i]];
				cx += float(p.x) / ((float)inliers->indices.size());
				cy += float(p.y) / ((float)inliers->indices.size());
				cz += float(p.z) / ((float)inliers->indices.size());
			}

			float ax, ay, az;
			float a11, a12, a13, a22, a23, a33;

			for (int i = 0; i < inliers->indices.size(); i++)
			{
				if (i == 0)
				{
					a11 = 0;
					a12 = 0;
					a13 = 0;
					a22 = 0;
					a23 = 0;
					a33 = 0;
				}

				pcl::PointXYZ p = pcdXYZ->points[inliers->indices[i]];

				ax = p.x - cx;
				ay = p.y - cy;
				az = p.z - cz;

				a11 += (ax * ax);
				a12 += (ax * ay);
				a13 += (ax * az);
				a22 += (ay * ay);
				a23 += (ay * az);
				a33 += (az * az);
			}

			a11 /= ((float)inliers->indices.size());
			a12 /= ((float)inliers->indices.size());
			a13 /= ((float)inliers->indices.size());
			a22 /= ((float)inliers->indices.size());
			a23 /= ((float)inliers->indices.size());
			a33 /= ((float)inliers->indices.size());

			cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
			cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
			cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));

			matA1.at<float>(0, 0) = a11;
			matA1.at<float>(0, 1) = a12;
			matA1.at<float>(0, 2) = a13;
			matA1.at<float>(1, 0) = a12;
			matA1.at<float>(1, 1) = a22;
			matA1.at<float>(1, 2) = a23;
			matA1.at<float>(2, 0) = a13;
			matA1.at<float>(2, 1) = a23;
			matA1.at<float>(2, 2) = a33;

			cv::eigen(matA1, matD1, matV1);

			float ratio = matD1.at<float>(0, 1) / matD1.at<float>(0, 0);

			Eigen::Vector3d normalToPlane;
			Eigen::Vector3d normalFromPlane(coeff[0], coeff[1], coeff[2]);

			if (cameraNum == 0)
				normalToPlane << 1.0, 0.0, 0.0;
			else if (cameraNum == 1)
				normalToPlane << cos(-72 / 180 * M_PI), sin(-72 / 180 * M_PI), 0.0;
			else if (cameraNum == 2)
				normalToPlane << cos(-144 / 180 * M_PI), sin(-144 / 180 * M_PI), 0.0;
			else if (cameraNum == 3)
				normalToPlane << cos(180 / 180 * M_PI), sin(180 / 180 * M_PI), 0.0;
			else if (cameraNum == 4)
				normalToPlane << cos(72 / 180 * M_PI), sin(72 / 180 * M_PI), 0.0;

			std::cout << "(" << iter << ")Num of inliers: " << inliers->indices.size() << std::endl;
			std::cout << "(" << iter << ")Coeff: " << coeff[0] << " " << coeff[1] << " " << coeff[2] << std::endl;
			std::cout << "(" << iter << ")Eigen value ratio: " << ratio << std::endl;
			std::cout << "(" << iter << ")Normal Check: " << abs(normalToPlane.dot(normalFromPlane)) << std::endl;

			if (ratio > 0.6 && abs(normalToPlane.dot(normalFromPlane)) > 0.5)
			{
				std::cout << "GGGGGGGGGGGGGGGGGGGGGGGGGGGGG@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;

				for (auto &p : *pcdXYZOrig)
				{
					double distToCenter = sqrt(pow(p.x - cx, 2.0) + pow(p.y - cy, 2.0) + pow(p.z - cz, 2.0));
					double distToPlane = abs(coeff[0] * p.x + coeff[1] * p.y + coeff[2] * p.z + coeff[3]);

					if (distToCenter < squareRange && distToPlane < maxDist && abs(coeff[2]) < 0.2)
					{
						pcl::PointXYZRGB p_rgb;

						pcdPlane.push_back(p);

						p_rgb.x = p.x;
						p_rgb.y = p.y;
						p_rgb.z = p.z;
						p_rgb.r = 0.0;
						p_rgb.g = 255.0;
						p_rgb.b = 0.0;

						pcdXYZRGB.push_back(p_rgb);
						pcdXYZCheckerboard->push_back(p);
					}

					else
					{
						pcl::PointXYZRGB p_rgb;

						p_rgb.x = p.x;
						p_rgb.y = p.y;
						p_rgb.z = p.z;
						p_rgb.r = 125.0;
						p_rgb.g = 125.0;
						p_rgb.b = 125.0;

						pcdXYZRGB2.push_back(p_rgb);
					}
				}

				planeParams(0, 0) = coeff[0];
				planeParams(1, 0) = coeff[1];
				planeParams(2, 0) = coeff[2];
				planeParams(3, 0) = coeff[3];

				break;
			}
		}

		// Extract inliers
		extract.setInputCloud(pcdXYZ);
		extract.setIndices(inliers);
		extract.setNegative(true);
		pcl::PointCloud<pcl::PointXYZ> cloudF;
		extract.filter(cloudF);

		pcdXYZ->swap(cloudF);

		iter += 1;

		if (pcdXYZ->points.size() < minInliers)
		{
			std::cout << "Break here!  " << pcdXYZ->points.size() << "//" << minInliers << std::endl;
			break;
		}
	}

	pcl::PointCloud<pcl::PointXYZ> pcdTransform;
	pcl::PointCloud<pcl::PointXYZ> pcdLine;

	bool flagCheckerboad = (pcdXYZRGB.size() > 0);

	if (pcdXYZRGB.size() > 0)
		std::cout << "Chekerboard plane detected. (" << pcdXYZRGB.size() << ")" << std::endl;

	// find lines !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	pcl::PointCloud<pcl::PointXYZ> pcdXYZforLine;
	if (flagCheckerboad)
	{

		double z_max, z_min;

		double a, b, c, d;
		a = planeParams(0, 0);
		b = planeParams(1, 0);
		c = planeParams(2, 0);
		d = planeParams(3, 0);

		for (int i = 0; i < pcdXYZRGB.size(); i++)
		{
			if (i == 0)
			{
				z_max = -1000000.0;
				z_min = 1000000.0;
			}
			pcl::PointXYZRGB p = pcdXYZRGB.points[i];

			// projection to plane
			double t = -(a * p.x + b * p.y + c * p.z + d) / ((a * a) + (b * b) + (c * c));

			p.x += a * t;
			p.y += b * t;
			p.z += c * t;

			pcdXYZRGB.points[i] = p;

			if (p.z > z_max)
				z_max = p.z;

			if (p.z < z_min)
				z_min = p.z;

			// distance filtering
			double dist = sqrt(pow(p.x - cx, 2) + pow(p.y - cy, 2) + pow(p.z - cz, 2));
			if (dist > 0.25)
			{
				pcl::PointXYZ pp;
				pp.x = p.x;
				pp.y = p.y;
				pp.z = p.z;
				pcdXYZforLine.push_back(pp);
			}
		}

		std::cout << "PCD for line is ready." << std::endl;

		double z0 = z_min + eplisonZ;
		double z1 = z_max - eplisonZ;

		double t = (z1 - z0) / ((double)numFrag);

		std::vector<std::vector<int>> ptsIdxOnLine(numFrag);

		Eigen::MatrixXd edgePoints(numFrag, 6);

		for (int i = 0; i < pcdXYZforLine.size(); i++)
		{
			int lineIdx = -1;
			double minDist2Line = 10000.0;
			for (int j = 0; j < numFrag; j++)
			{
				double zLine = z0 + j * t;
				double dist2Line = abs(zLine - pcdXYZforLine.points[i].z);
				if (dist2Line < minDist2Line)
				{
					minDist2Line = dist2Line;
					lineIdx = j;
				}
			}

			if (minDist2Line < maxDistLine)
			{
				// pcdLine.push_back(pcdXYZforLine.points[i]);
				ptsIdxOnLine[lineIdx].push_back(i);
			}
		}

		std::cout << "ptsIdxOnLine is ready." << std::endl;

		std::vector<int> addTarget;

		std::vector<std::vector<int>> ptsIdxOnLineD;

		for (int i = 0; i < ptsIdxOnLine.size(); i++)
		{
			if (ptsIdxOnLine[i].size() > 5)
				addTarget.push_back(i);

			// std::cout << "ptsIdxOnLine[] size: " << ptsIdxOnLine[i].size() << std::endl;
		}

		for (int i = 0; i < addTarget.size(); i++)
			ptsIdxOnLineD.push_back(ptsIdxOnLine[addTarget[i]]);

		ptsIdxOnLine = ptsIdxOnLineD;

		// Find outer edge point of checkerboard!
		for (int i = 0; i < ptsIdxOnLine.size(); i++)
		{
			double cxl, cyl, czl;

			for (int j = 0; j < ptsIdxOnLine[i].size(); j++)
			{
				if (j == 0)
				{
					cxl = 0;
					cyl = 0;
					czl = 0;
				}
				// find center first
				cxl += (pcdXYZforLine[ptsIdxOnLine[i][j]].x / ((double)ptsIdxOnLine[i].size()));
				cyl += (pcdXYZforLine[ptsIdxOnLine[i][j]].y / ((double)ptsIdxOnLine[i].size()));
				czl += (pcdXYZforLine[ptsIdxOnLine[i][j]].z / ((double)ptsIdxOnLine[i].size()));
			}

			// calc dist
			std::vector<double> dist2center(ptsIdxOnLine[i].size());
			for (int j = 0; j < ptsIdxOnLine[i].size(); j++)
			{
				double x = pcdXYZforLine[ptsIdxOnLine[i][j]].x;
				double y = pcdXYZforLine[ptsIdxOnLine[i][j]].y;
				double z = pcdXYZforLine[ptsIdxOnLine[i][j]].z;

				double dist = sqrt(pow(x - cxl, 2) + pow(y - cyl, 2) + pow(z - czl, 2));
				dist2center[j] = dist;
			}

			std::vector<int> order(ptsIdxOnLine[i].size());
			std::iota(order.begin(), order.end(), 0);
			std::sort(order.begin(), order.end(), [&](int i, int j)
					  { return dist2center[i] < dist2center[j]; });

			std::reverse(order.begin(), order.end());

			Eigen::Vector3d vec1 = {pcdXYZforLine[ptsIdxOnLine[i][order[0]]].x - cxl,
									pcdXYZforLine[ptsIdxOnLine[i][order[0]]].y - cyl,
									pcdXYZforLine[ptsIdxOnLine[i][order[0]]].z - czl};

			bool findFlag = false;

			for (int j = 1; j < order.size(); j++)
			{
				Eigen::Vector3d vec2 = {pcdXYZforLine[ptsIdxOnLine[i][order[j]]].x - cxl,
										pcdXYZforLine[ptsIdxOnLine[i][order[j]]].y - cyl,
										pcdXYZforLine[ptsIdxOnLine[i][order[j]]].z - czl};

				double inner = (vec1 / vec1.norm()).dot(vec2 / vec2.norm());

				if (inner < -0.5)
				{
					pcdLine.push_back(pcdXYZforLine[ptsIdxOnLine[i][order[0]]]);
					pcdLine.push_back(pcdXYZforLine[ptsIdxOnLine[i][order[j]]]);

					edgePoints(i, 0) = vec1[0];
					edgePoints(i, 1) = vec1[1];
					edgePoints(i, 2) = vec1[2];
					edgePoints(i, 3) = vec2[0];
					edgePoints(i, 4) = vec2[1];
					edgePoints(i, 5) = vec2[2];
					findFlag = true;

					break;
				}
			}

			if (!findFlag)
			{
				std::cout << "WTF???????????????????????????????????????????????" << std::endl;
			}
		}

		std::cout << "edgePoints is ready." << std::endl;

		// find four lines

		// line fitting
		pcl::ModelCoefficients::Ptr coefficientsL(new pcl::ModelCoefficients);
		pcl::PointIndices::Ptr inliersL(new pcl::PointIndices);
		pcl::SACSegmentation<pcl::PointXYZ> segL;
		pcl::ExtractIndices<pcl::PointXYZ> extractL;
		segL.setOptimizeCoefficients(true);
		segL.setModelType(pcl::SACMODEL_LINE);
		segL.setMethodType(pcl::SAC_RANSAC);
		segL.setDistanceThreshold(maxDistLine);

		int maxIter = 6;
		int iter = 0;
		int numLines = 0;
		std::vector<std::vector<float>> lineParams;
		while (true)
		{
			segL.setInputCloud(pcdLine.makeShared());
			segL.segment(*inliersL, *coefficientsL);

			std::vector<float> coeffL = coefficientsL->values;

			if (inliersL->indices.size() >= numInliersLine)
			{
				for (int i = 0; i < inliersL->indices.size(); i++)
				{
					pcl::PointXYZ p = pcdLine.points[inliersL->indices[i]];
					pcdLine2.push_back(p);
				}
				numLines += 1;
				lineParams.push_back(coeffL);
			}

			extractL.setInputCloud(pcdLine.makeShared());
			extractL.setIndices(inliersL);
			extractL.setNegative(true);
			pcl::PointCloud<pcl::PointXYZ> cloudEx;
			extractL.filter(cloudEx);
			pcdLine.swap(cloudEx);

			if (pcdLine.points.size() < numInliersLine || inliersL->indices.size() == 0)
				break;

			iter += 1;
			if (iter >= maxIter)
				break;
		}

		bool flagLine = (numLines == 4);

		std::cout << "Four line found?: " << numLines << std::endl;

		if (flagLine)
		{
			Eigen::Vector3f d1;
			d1 << lineParams[0][3], lineParams[0][4], lineParams[0][5];

			int idxParallel = -1;
			for (int i = 1; i < 4; i++)
			{

				Eigen::Vector3f d2;
				d2 << lineParams[i][3], lineParams[i][4], lineParams[i][5];

				if (abs(d1.dot(d2)) > 0.9)
				{
					idxParallel = i;
					break;
				}
			}

			std::vector<std::vector<float>> lineParamsTemp(2);
			lineParamsTemp[0] = lineParams[0];

			if (idxParallel != -1)
			{

				lineParamsTemp[1] = lineParams[idxParallel];

				for (int i = 1; i < 4; i++)
				{
					if (i != idxParallel)
					{
						lineParamsTemp.push_back(lineParams[i]);
					}
				}

				lineParams = lineParamsTemp;

				Eigen::MatrixXf X(4, 2);

				std::vector<std::vector<int>> combinations = {{0, 2}, {0, 3}, {1, 2}, {1, 3}};

				bool isValid = false;

				for (int i = 0; i < 4; i++)
				{

					// intersection point between line[ii] and line[jj]
					int ii = combinations[i][0];
					int jj = combinations[i][1];

					Eigen::MatrixXf A(3, 2);
					Eigen::MatrixXf b(3, 1);
					Eigen::MatrixXf x(2, 1);

					A << lineParams[ii][3], -lineParams[jj][3],
						lineParams[ii][4], -lineParams[jj][4],
						lineParams[ii][5], -lineParams[jj][5];

					b << lineParams[jj][0] - lineParams[ii][0],
						lineParams[jj][1] - lineParams[ii][1],
						lineParams[jj][2] - lineParams[ii][2];

					Eigen::MatrixXf At = A.transpose();

					x = ((At * A).inverse()) * (At * b);
					X.block(i, 0, 1, 2) = x.transpose();

					isValid = (abs(x(0, 0)) < 1.0 && abs(x(1, 0)) < 1.0);

					if (!isValid)
					{
						break;
					}
				}

				if (isValid)
				{
					// std::cout << "Valid Observation" << std::endl;
					// std::cout << X << std::endl;

					Eigen::Vector3f normalPlane;
					normalPlane << coeff[0], coeff[1], coeff[2];
					normalPlane /= normalPlane.norm();

					pcl::PointCloud<pcl::PointXYZ> pcdEdgeTemp;

					float xc, yc, zc;
					float x1, y1, z1, a1, b1, c1, x2, y2, z2, a2, b2, c2;

					for (int i = 0; i < 4; i++)
					{
						int ii = combinations[i][0];
						int jj = combinations[i][1];

						if (i == 0)
						{
							xc = 0.0;
							yc = 0.0;
							zc = 0.0;
						}

						float lambda = X(i, 0);
						float mu = X(i, 1);
						x1 = lineParams[ii][0];
						y1 = lineParams[ii][1];
						z1 = lineParams[ii][2];
						a1 = lineParams[ii][3];
						b1 = lineParams[ii][4];
						c1 = lineParams[ii][5];

						x2 = lineParams[jj][0];
						y2 = lineParams[jj][1];
						z2 = lineParams[jj][2];
						a2 = lineParams[jj][3];
						b2 = lineParams[jj][4];
						c2 = lineParams[jj][5];
						Eigen::Vector3f p1;
						p1 << x1 + a1 * lambda, y1 + b1 * lambda, z1 + c1 * lambda;
						Eigen::Vector3f p2;
						p2 << x2 + a2 * mu, y2 + b2 * mu, z2 + c2 * mu;

						Eigen::Vector3f pIntersect = (p1 + p2) / 2.0;

						pcl::PointXYZ pI;
						pI.x = pIntersect(0);
						pI.y = pIntersect(1);
						pI.z = pIntersect(2);

						xc += pIntersect(0) / 4.0;
						yc += pIntersect(1) / 4.0;
						zc += pIntersect(2) / 4.0;

						pcdEdgeTemp.push_back(pI);
					}

					pcdEdge = pcdEdgeTemp;

					// normal direction check
					Eigen::Vector3f vecPc;
					vecPc << xc, yc, zc;
					vecPc /= vecPc.norm();

					// std::cout << "Normal check: " << normalPlane.dot(vecPc) << std::endl;
					if (normalPlane.dot(vecPc) > 0)
						normalPlane *= -1.0;

					// find order
					double zMin = 100000.0;
					double zMax = -100000.0;
					int idxP0, idxP1, idxP2, idxP3;

					for (int i = 0; i < 4; i++)
					{
						if (pcdEdge[i].z > zMax)
						{
							idxP0 = i;
							zMax = pcdEdge[i].z;
						}

						if (pcdEdge[i].z < zMin)
						{
							idxP2 = i;
							zMin = pcdEdge[i].z;
						}
					}

					Eigen::Vector3f p0;
					p0 << pcdEdge[idxP0].x, pcdEdge[idxP0].y, pcdEdge[idxP0].z;

					Eigen::Vector3f p2;
					p2 << pcdEdge[idxP2].x, pcdEdge[idxP2].y, pcdEdge[idxP2].z;

					for (int i = 0; i < 4; i++)
					{
						if (i != idxP0 && i != idxP2)
						{
							Eigen::Vector3f px;
							px << pcdEdge[i].x, pcdEdge[i].y, pcdEdge[i].z;

							Eigen::Vector3f vecCross = ((px - p0).cross(p2 - px));
							vecCross /= vecCross.norm();
							if (vecCross.dot(normalPlane) > 0)
								idxP1 = i;
							else
								idxP3 = i;
						}
					}

					Eigen::Vector3f p1;
					p1 << pcdEdge[idxP1].x, pcdEdge[idxP1].y, pcdEdge[idxP1].z;

					Eigen::Vector3f p3;
					p3 << pcdEdge[idxP3].x, pcdEdge[idxP3].y, pcdEdge[idxP3].z;

					// check is square?

					double inner01 = abs(((p1 - p0) / ((p1 - p0).norm())).dot(((p2 - p1) / ((p2 - p1).norm()))));
					double inner12 = abs(((p2 - p1) / ((p2 - p1).norm())).dot(((p3 - p2) / ((p3 - p2).norm()))));
					double inner23 = abs(((p3 - p2) / ((p3 - p2).norm())).dot(((p0 - p3) / ((p0 - p3).norm()))));
					double inner30 = abs(((p0 - p3) / ((p0 - p3).norm())).dot(((p1 - p0) / ((p1 - p0).norm()))));

					double thresh = 0.05;
					bool isSquare = (inner01 < thresh && inner12 < thresh && inner23 < thresh && inner30 < thresh);

					if (isSquare)
					{

						// check index

						// std::cout << "Index of P0 ~ P3: " << idxP0 << " / " << idxP1 << " / " << idxP2 << " / " << idxP3 << std::endl;

						std::vector<std::vector<double>> colorRGBW = {{255.0, 0, 0}, {0, 255.0, 0}, {0, 0, 255.0}, {255.0, 255.0, 255.0}};
						std::vector<int> idxs = {idxP0, idxP1, idxP2, idxP3};

						for (int i = 0; i < 4; i++)
						{
							pcl::PointXYZRGB pc;
							pc.x = pcdEdge[idxs[i]].x;
							pc.y = pcdEdge[idxs[i]].y;
							pc.z = pcdEdge[idxs[i]].z;
							pc.r = colorRGBW[i][0];
							pc.g = colorRGBW[i][1];
							pc.b = colorRGBW[i][2];

							pcdEdgeColor.push_back(pc);

							// Visualize line

							geometry_msgs::Point p1, p2;
							p1.x = pcdEdge[idxs[i]].x;
							p1.y = pcdEdge[idxs[i]].y;
							p1.z = pcdEdge[idxs[i]].z;
							p2.x = pcdEdge[idxs[(i + 1) % 4]].x;
							p2.y = pcdEdge[idxs[(i + 1) % 4]].y;
							p2.z = pcdEdge[idxs[(i + 1) % 4]].z;

							visualization_msgs::Marker line;
							line.type = visualization_msgs::Marker::LINE_STRIP;
							line.ns = "checkerLine";
							line.id = i;
							line.action = visualization_msgs::Marker::ADD;
							line.scale.x = 0.05;
							line.color.r = 0.0;
							line.color.g = 1.0;
							line.color.b = 0.0;
							line.color.a = 1;
							line.points.push_back(p1);
							line.points.push_back(p2);

							line.header.frame_id = "lidar";
							line.lifetime = ros::Duration(0.3);
							checkerLine.markers.push_back(line);
						}

						Eigen::Vector3f v1 = {pcdEdge[idxs[3]].x - pcdEdge[idxs[0]].x,
											  pcdEdge[idxs[3]].y - pcdEdge[idxs[0]].y,
											  pcdEdge[idxs[3]].z - pcdEdge[idxs[0]].z};

						Eigen::Vector3f v2 = {pcdEdge[idxs[1]].x - pcdEdge[idxs[0]].x,
											  pcdEdge[idxs[1]].y - pcdEdge[idxs[0]].y,
											  pcdEdge[idxs[1]].z - pcdEdge[idxs[0]].z};

						float initialCol = v1.norm() * (6.0 / 100.0);
						float stepCol = v1.norm() * (8.0 / 100.0);

						float initialRow = v2.norm() * (6.0 / 100.0);
						float stepRow = v2.norm() * (8.0 / 100.0);

						float x0 = pcdEdge[idxs[0]].x + initialCol * v1(0) + initialRow * v2(0);
						float y0 = pcdEdge[idxs[0]].y + initialCol * v1(1) + initialRow * v2(1);
						float z0 = pcdEdge[idxs[0]].z + initialCol * v1(2) + initialRow * v2(2);

						for (int row = 0; row < 10; row++)
						{
							for (int col = 0; col < 12; col++)
							{
								// pcdCheckerEdgeColor
								pcl::PointXYZRGB pc;
								pc.x = x0 + ((float)col * stepCol) * v1(0) + ((float)row * stepRow) * v2(0);
								pc.y = y0 + ((float)col * stepCol) * v1(1) + ((float)row * stepRow) * v2(1);
								pc.z = z0 + ((float)col * stepCol) * v1(2) + ((float)row * stepRow) * v2(2);

								if (col == 0 || row == 0 || col == 11 || row == 9)
								{
									pc.r = 0.0;
									pc.g = 255.0;
									pc.b = 0.0;
								}
								else
								{
									pc.r = 255.0;
									pc.g = 0.0;
									pc.b = 0.0;
								}

								pcdCheckerEdgeColor.push_back(pc);

								if (col > 0 && col < 11 && row > 0 && row < 9)
								{
									Eigen::Vector3f pvec(pc.x, pc.y, pc.z);
									pointsChecker.push_back(pvec);
								}
							}
						}

						pointsCheckerTotal.push_back(pointsChecker);
						cornersTotal.push_back(corners);
					}
					else
					{
						std::cout << "########################## Not Square!!!!!!!!!!!!! ##########################" << std::endl;
						std::cout << "Inner 01: " << inner01 << std::endl;
						std::cout << "Inner 12: " << inner12 << std::endl;
						std::cout << "Inner 23: " << inner23 << std::endl;
						std::cout << "Inner 30: " << inner30 << std::endl;
					}
				}
			}
		}
	}

	// std::cout << "DEBUG 4" << std::endl;
	// bool isReadyToVis = (pointsChecker.size() > 0);

	// if (isReadyToVis)
	// {

	// 	// projection matrix
	// 	Eigen::MatrixXf P = K * Tcl;

	// 	for (int i = 0; i < pointsChecker.size(); i++)
	// 	{
	// 		Eigen::Vector4f phat(pointsChecker[i][0], pointsChecker[i][1], pointsChecker[i][2], 1.0);
	// 		Eigen::Vector3f xys = P * phat;
	// 		xys /= xys(2);
	// 		cv::line(cv_ptr->image, cv::Point2f(xys(0), xys(1)), corners[i], cv::Scalar((int)0, (int)255, (int)0), 3, 8, 0);
	// 		cv::circle(cv_ptr->image, cv::Point2f(xys(0), xys(1)), 10, cv::Scalar((int)0, (int)0, (int)255), -1, 8, 0);
	// 		cv::circle(cv_ptr->image, corners[i], 10, cv::Scalar((int)255, (int)0, (int)0), -1, 8, 0);
	// 	}

	// 	for (int i = 0; i < pcdXYZRGB.size(); i++)
	// 	{
	// 		Eigen::Vector4f phat(pcdXYZRGB[i].x, pcdXYZRGB[i].y, pcdXYZRGB[i].z, 1.0);
	// 		Eigen::Vector3f xys = P * phat;
	// 		xys /= xys(2);
	// 		cv::Vec3b rgb = cv_ptr->image.at<cv::Vec3b>(xys[1], xys[0]);
	// 		// cv::circle(cv_ptr->image, cv::Point2f(xys(0), xys(1)), 10, cv::Scalar((int)0, (int)255, (int)0), -1, 8, 0);
	// 		pcdXYZRGB[i].r = (double)rgb[0];
	// 		pcdXYZRGB[i].g = (double)rgb[1];
	// 		pcdXYZRGB[i].b = (double)rgb[2];
	// 	}

	// 	// convert from pcl point cloud to ros message
	// 	sensor_msgs::PointCloud2 thisLaserCloud;
	// 	pcl::toROSMsg(pcdXYZRGB, thisLaserCloud);
	// 	thisLaserCloud.header.frame_id = "/lidar";

	// 	pubLaserCloud.publish(thisLaserCloud);
	// }

	std::cout << "The number of optimization stack: " << pointsCheckerTotal.size() << std::endl;

	bool isReadyToOpt = (pointsCheckerTotal.size() > 30);

	if (isReadyToOpt)
	{
		std::cout << "Optimize @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
		Eigen::Matrix3f Rcl = Tcl.block(0, 0, 3, 3);
		Eigen::Vector3d tcl((double)Tcl(0, 3), (double)Tcl(1, 3), (double)Tcl(2, 3));
		// Eigen::Quaternionf qclEigen(Rcl);
		float qw = 0.5 * sqrt(1 + Rcl(0, 0) + Rcl(1, 1) + Rcl(2, 2));
		float qx = 0.25 / qw * (Rcl(2, 1) - Rcl(1, 2));
		float qy = 0.25 / qw * (Rcl(0, 2) - Rcl(2, 0));
		float qz = 0.25 / qw * (Rcl(1, 0) - Rcl(0, 1));
		Eigen::Vector4d qcl((double)qw, (double)qx, (double)qy, (double)qz);
		Eigen::Vector4d qclOrig((double)qw, (double)qx, (double)qy, (double)qz);

		ceres::Problem problem;
		ceres::Solver::Options options;
		ceres::Solver::Summary summary;

		std::cout << "Value change check &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" << std::endl;
		std::cout << "Trans  (before): " << tcl.transpose() << std::endl;
		std::cout << "Rot(q) (before): " << qcl.transpose() << " " << std::endl;

		options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
		options.minimizer_progress_to_stdout = false;
		options.trust_region_strategy_type = ceres::DOGLEG;
		options.num_threads = 10;
		options.max_num_iterations = 100;
		// options.max_solver_time_in_seconds = 0.04;

		problem.AddParameterBlock(qcl.data(), 4);
		problem.AddParameterBlock(tcl.data(), 3);

		for (int i = 0; i < pointsCheckerTotal.size(); i++)
		{
			for (int j = 0; j < pointsCheckerTotal[i].size(); j++)
			{
				Eigen::Vector2f xy(cornersTotal[i][j].x, cornersTotal[i][j].y);
				Eigen::Vector3f pw = pointsCheckerTotal[i][j];

				ceres::CostFunction *cost_function = ProjectionFactor::Create(xy, pw, K);
				problem.AddResidualBlock(cost_function, new ceres::HuberLoss(0.5), qcl.data(), tcl.data());
			}
		}

		ceres::Solve(options, &problem, &summary);

		std::cout << "..................................................." << std::endl;
		std::cout << "Trans  (after): " << tcl.transpose() << std::endl;
		qcl /= qcl.norm();
		std::cout << "Rot(q) (after): " << qcl.transpose() << std::endl;

		std::cout << summary.BriefReport() << std::endl;

		Eigen::Matrix3f Ropt;

		// qcl = qclOrig;

		// Q --> R
		Ropt(0, 0) = (float)(2 * (qcl[0] * qcl[0] + qcl[1] * qcl[1]) - 1);
		Ropt(0, 1) = (float)(2 * (qcl[1] * qcl[2] - qcl[0] * qcl[3]));
		Ropt(0, 2) = (float)(2 * (qcl[1] * qcl[3] + qcl[0] * qcl[2]));
		Ropt(1, 0) = (float)(2 * (qcl[1] * qcl[2] + qcl[0] * qcl[3]));
		Ropt(1, 1) = (float)(2 * (qcl[0] * qcl[0] + qcl[2] * qcl[2]) - 1);
		Ropt(1, 2) = (float)(2 * (qcl[2] * qcl[3] - qcl[0] * qcl[1]));
		Ropt(2, 0) = (float)(2 * (qcl[1] * qcl[3] - qcl[0] * qcl[2]));
		Ropt(2, 1) = (float)(2 * (qcl[2] * qcl[3] + qcl[0] * qcl[1]));
		Ropt(2, 2) = (float)(2 * (qcl[0] * qcl[0] + qcl[3] * qcl[3]) - 1);

		std::cout << "Rot CHECK" << std::endl;
		std::cout << Ropt << std::endl;

		// Tcl.block(0, 0, 3, 3) = Ropt;
		// Tcl(0, 3) = (float)tcl[0];
		// Tcl(1, 3) = (float)tcl[1];
		// Tcl(2, 3) = (float)tcl[2];

		std::cout << "### End Optimize @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
	}

	pubMarkerArray.publish(checkerLine);

	// segmentation target
	sensor_msgs::PointCloud2 thisLaserCloud6;
	pcl::toROSMsg(pcdSegTarget, thisLaserCloud6);
	thisLaserCloud6.header.frame_id = "/lidar";

	pubDebug.publish(thisLaserCloud6);

	// all plane points
	sensor_msgs::PointCloud2 thisLaserCloud7;
	pcl::toROSMsg(pcdPlane, thisLaserCloud7);
	thisLaserCloud7.header.frame_id = "/lidar";

	pubPlaneAll.publish(thisLaserCloud7);

	// sensor_msgs::PointCloud2 thisLaserCloud2;
	// pcl::toROSMsg(pcdXYZforLine, thisLaserCloud2);
	// thisLaserCloud2.header.frame_id = "/lidar";

	// pubTransformedCloud.publish(thisLaserCloud2);

	sensor_msgs::PointCloud2 thisLaserCloud3;
	pcl::toROSMsg(pcdLine2, thisLaserCloud3);
	thisLaserCloud3.header.frame_id = "/lidar";

	pubLineCloud.publish(thisLaserCloud3);

	sensor_msgs::PointCloud2 thisLaserCloud4;
	pcl::toROSMsg(pcdEdgeColor, thisLaserCloud4);
	thisLaserCloud4.header.frame_id = "/lidar";

	pubLineEdge.publish(thisLaserCloud4);

	sensor_msgs::PointCloud2 thisLaserCloud5;
	pcl::toROSMsg(pcdCheckerEdgeColor, thisLaserCloud5);
	thisLaserCloud5.header.frame_id = "/lidar";

	pubCheckerEdge.publish(thisLaserCloud5);

	// std::vector<std::vector<double>> colors = {{255.0, 0.0, 0.0}, {0.0, 255.0, 0.0}, {0.0, 0.0, 255.0}};
	// for(int i=0; i<15; i++){
	// 	std::vector<double> color = colors[i%3];
	// 	cv::circle(cv_ptr->image, corners[i], 10, cv::Scalar((int)color[0], (int)color[1], (int)color[2]), 1, 8, 0);
	// }

	// cv::drawChessboardCorners(cv_ptr->image, patternSize, cv::Mat(corners), patternFound);

	// convert from cv img to ros msg
	sensor_msgs::Image::Ptr thisImgRectifiedPtr = (cv_ptr->toImageMsg());
	// imgQueue.push_back(*thisImgRectifiedPtr);

	// publish test
	pubImg.publish(thisImgRectifiedPtr);
}

void LidarCamCalib::imgCloudSyncCallback(const sensor_msgs::ImageConstPtr &imgMsg, const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{
	// std::cout << "LIDAR TIME: " << std::setprecision(18) << static_cast<double>(laserCloudMsg->header.stamp.toSec()) << std::endl;
	// std::cout << "IMAGE TIME: " << std::setprecision(18) << static_cast<double>(imgMsg->header.stamp.toSec()) << std::endl;

	sensorDataHandler(imgMsg, laserCloudMsg);
}

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> syncPolicy;

int main(int argc, char **argv)
{
	std::cout << "YEA! START!" << std::endl;

	ros::init(argc, argv, "lidar_cam_calib");
	ros::NodeHandle nh;

	LidarCamCalib calib;

	message_filters::Subscriber<sensor_msgs::Image> imgSubSync(nh, calib.imgTopic, 40);
	message_filters::Subscriber<sensor_msgs::PointCloud2> laserCloudSubSync(nh, calib.laserCloudTopic, 40);

	message_filters::Synchronizer<syncPolicy> sync(syncPolicy(10), imgSubSync, laserCloudSubSync);
	sync.registerCallback(boost::bind(&LidarCamCalib::imgCloudSyncCallback, &calib, _1, _2));

	ros::spin();
	return 0;
}
