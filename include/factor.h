#ifndef FACTOR_H
#define FACTOR_H

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <cmath>
class SquareFactor
{
public:
    explicit SquareFactor(const double &x, const double &y, const double &cx, const double &cy) : x_(x), y_(y), cx_(cx), cy_(cy) {}

    static ceres::CostFunction *Create(const double &x, const double &y, const double &cx, const double &cy)
    {
        return (new ceres::AutoDiffCostFunction<SquareFactor, 4, 3>(
            new SquareFactor(x, y, cx, cy)));
    }

    template <typename T>
    bool operator()(const T *const RT, T *residuals) const
    {

        T tx = RT[0];
        T ty = RT[1];
        T theta = RT[2];

        T x_hat = cos(theta) * x_ + sin(theta) * y_ - tx;
        T y_hat = -sin(theta) * x_ + cos(theta) * y_ - ty;

        residuals[0] = x_hat * x_hat - T(0.25);
        residuals[1] = y_hat * y_hat - T(0.25);
        residuals[2] = T(cx_) - tx;
        residuals[3] = T(cy_) - ty;
        return true;
    }

private:
    const double x_;
    const double y_;
    const double cx_;
    const double cy_;
};

// Standard bundle adjustment cost function for variable
// camera pose and calibration and point parameters.

class ProjectionFactor
{
public:
    explicit ProjectionFactor(const Eigen::Vector2f &point2D, const Eigen::Vector3f &point3D, const Eigen::Matrix3f &intrinsic) : point2D_(point2D),
                                                                                                                                  point3D_(point3D),
                                                                                                                                  intrinsic_(intrinsic) {}

    static ceres::CostFunction *Create(const Eigen::Vector2f &point2D, const Eigen::Vector3f &point3D, const Eigen::Matrix3f &intrinsic)
    {
        return (new ceres::AutoDiffCostFunction<ProjectionFactor, 2, 4, 3>(
            new ProjectionFactor(point2D, point3D, intrinsic)));
    }
 
    template <typename T>
    bool operator()(const T *const qvec, const T *const tvec, T *residuals) const
    {
        // Rotate and translate.
        T projection[3];
        T point3D[3];
        point3D[0] = T(point3D_[0]);
        point3D[1] = T(point3D_[1]);
        point3D[2] = T(point3D_[2]);

        //// R * P + t / w x y z
        ceres::QuaternionRotatePoint(qvec, point3D, projection);
        projection[0] += tvec[0];
        projection[1] += tvec[1];
        projection[2] += tvec[2];

        // Project to image plane.
        projection[0] /= projection[2];
        projection[1] /= projection[2];

        // Distort and transform to pixel space.
        // World To Image

        T fx = T(intrinsic_(0, 0));
        T fy = T(intrinsic_(1, 1));
        T cx = T(intrinsic_(0, 2));
        T cy = T(intrinsic_(1, 2));

        // No distortion
        residuals[0] = (fx * projection[0] + cx) - T(point2D_[0]);
        residuals[1] = (fy * projection[1] + cy) - T(point2D_[1]);
        // std::cerr<<"Check Residual: "<<residuals[0]<<" / "<<residuals[1]<<std::endl;

        return true;
    }

private:
    const Eigen::Vector2f point2D_;
    const Eigen::Vector3f point3D_;
    const Eigen::Matrix3f intrinsic_;
};

#endif