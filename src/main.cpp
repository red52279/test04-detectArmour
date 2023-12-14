#include <opencv2/opencv.hpp>
#include <detect/detect.h>

cv::Scalar aim_color_red(15, 0.5, 0.5);
cv::Scalar aim_color_range_red(10,0.5,0.5);
cv::Scalar red(0,0,255);

cv::Scalar aim_color_blue(175, 0.32, 0.8);
cv::Scalar aim_color_range_blue(175,0.32,0.2);
cv::Scalar blue(255, 0, 0);

cv::Mat Erode_Kernel = cv::getStructuringElement(0, cv::Size(3, 3));
cv::Mat Dilate_Kernel = cv::getStructuringElement(0, cv::Size(3, 3));

int main() {
    cv::VideoCapture cap("../source/RD0_Trim.mp4");
    cv::Mat cap_img;
    while(true)
    {
        cap.read(cap_img);
        cv::Mat preprocessed_img = detect::preprocess(cap_img, aim_color_blue, aim_color_range_blue,
                                              Erode_Kernel, Dilate_Kernel, 1, 2);
        cv::Mat img_canny = detect::detectContours(preprocessed_img, cap_img, red, true);

        cv::namedWindow("img", cv::WINDOW_NORMAL);
        cv::imshow("img", cap_img);

        int c = cv::waitKey(1);
        if((c & 255) == 27) break;
    }

    cv::waitKey(0);

    return 0;
}