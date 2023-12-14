#ifndef _DETECT_H_
#define _DETECT_H_

#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

namespace detect {
    cv::Mat preprocess(cv::Mat& img, cv::Scalar& AimColor, cv::Scalar& ColorRange,
                       cv::Mat& ErodeKernel, cv::Mat& DilateKernel,
                       int ErodeTimes, int DilateTimes);

    cv::Mat FindImgContours(cv::Mat &img, int CannyLowThresh = 100, int CannyHighThresh = 200);

    void detectArmour(cv::Mat &img, cv::Scalar &Color, int areaThread = 0);

    cv::Mat detectContours(cv::Mat &img, cv::Mat &draw_img,
                                                       cv::Scalar &AimColor, bool isDrawRect = true);

}
#endif
