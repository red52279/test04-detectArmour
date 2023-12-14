#include <detect/detect.h>


std::vector<std::vector<cv::Point>> contours;
std::vector<cv::Vec4i> hierarchy;

struct line
{
    float k;
    float length;
    cv::Point2f center_point;
    cv::Point2f one_endpoint;
    cv::Point2f other_endpoint;
};

struct armour
{
    float radius;
    cv::Point2f center_point;
};

namespace detect {
    cv::Mat preprocess(cv::Mat &img, cv::Scalar &AimColor, cv::Scalar &ColorRange,
                       cv::Mat &ErodeKernel, cv::Mat &DilateKernel,
                       int ErodeTimes, int DilateTimes) {
        cv::Mat img32;
        cv::Scalar HSVL, HSVH;
        HSVL = AimColor - ColorRange;
        HSVH = AimColor + ColorRange;
        img.convertTo(img32, CV_32FC3, 1 / 255.0, 0);
        cv::cvtColor(img32, img32, cv::COLOR_BGR2HSV);
        cv::inRange(img32, HSVL, HSVH, img32);
        for (int i = 0; i < ErodeTimes; i++)
            cv::morphologyEx(img32, img32, cv::MORPH_ERODE, ErodeKernel);
        for (int i = 0; i < DilateTimes; i++)
            cv::morphologyEx(img32, img32, cv::MORPH_DILATE, DilateKernel);
        cv::copyMakeBorder(img32, img32, 1, 1, 1, 1, 0, cv::Scalar(0));
        return img32;
    }

    cv::Mat FindImgContours(cv::Mat &img, int CannyLowThresh, int CannyHighThresh) {
        cv::Mat imgCanny = cv::Mat::zeros(img.size(), CV_8U);
        cv::Canny(img, imgCanny, CannyLowThresh, CannyHighThresh);
        cv::findContours(imgCanny, contours, hierarchy, 0, 2, cv::Point());
        return imgCanny;
    }

    void detectArmour(cv::Mat &img, cv::Scalar &Color, int areaThread)
    {
        double area;

        line lines[contours.size() + 2];
        int line_count = 0;

        for (int i = 0; i < contours.size(); i++)
        {
            area = cv::contourArea(contours[i]);
            if (area < areaThread)
                continue;
            cv::RotatedRect rect = cv::minAreaRect(contours[i]);
            cv::Point2f points[4];
            rect.points(points);

            float x1, y1, x2, y2, x3, y3, x4, y4;        //4个中点坐标
            x1 = (points[0].x + points[1].x) / 2;
            y1 = (points[0].y + points[1].y) / 2;
            x2 = (points[2].x + points[3].x) / 2;
            y2 = (points[2].y + points[3].y) / 2;
            x3 = (points[0].x + points[3].x) / 2;
            y3 = (points[0].y + points[3].y) / 2;
            x4 = (points[1].x + points[2].x) / 2;
            y4 = (points[1].y + points[2].y) / 2;
            float line_width = sqrt((x3 - x4) * (x3 - x4) + (y3 - y4) * (y3 - y4));
            float line_height = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
            float line_k = 20.0;
            cv::Point2f line_point1;
            cv::Point2f line_point2;
            if (line_height < line_width) {
                float temp = line_height;
                line_height =  line_width;
                line_width = temp;
                if(x3 != x4)
                   line_k = (y4 - y3) / (x4 - x3);
                line_point1 = {x3, y3};
                line_point2 = {x4, y4};
            } else {
                if(x1 != x2)
                    line_k = (y2 - y1) / (x2 - x1);
                line_point1 = {x1, y1};
                line_point2 = {x2, y2};
            }
            float line_height_per_width = line_height / line_width;
            if(abs(line_k) < 2)
            {
                continue;
            }

            lines[line_count].one_endpoint = line_point1;
            lines[line_count].other_endpoint = line_point2;
            lines[line_count].center_point = (line_point1 + line_point2) / 2;
            lines[line_count].k = line_k;
            lines[line_count++].length = line_height;

        }

        armour armours[line_count + 2];
        int armour_count = 0;
        float line_distance1;
        float line_distance2;
        for(int i = 0; i < line_count; i++) {
            for(int j = i + 1; j < line_count; j++)
            {
                line_distance1 = lines[i].one_endpoint.x - lines[j].one_endpoint.x;
                line_distance2 = lines[i].other_endpoint.x - lines[j].other_endpoint.x;
                if( abs(abs(atan(lines[i].k)) - abs(atan(lines[j].k)) ) < 0.08 &&
                abs(lines[i].length - lines[j].length) < 50 &&
                        abs( ( (lines[i].length + lines[j].length) / 2) /
                                sqrt(line_distance1 * line_distance1 + line_distance2 * line_distance2)
                            ) > 0.28  )
                {
                    armours[armour_count].radius = (lines[i].length + lines[j].length) / 2;
                    armours[armour_count++].center_point = (lines[i].center_point + lines[j].center_point) / 2;
                }
            }
        }

        for(int i = 0; i < armour_count; i++)
        {
            cv::circle(img, armours[i].center_point, (int)armours[i].radius , Color, 2, 8, 0);
        }

    }

    cv::Mat detectContours(cv::Mat &img, cv::Mat &draw_img,
                                                       cv::Scalar &AimColor, bool isDrawRect) {
        cv::Mat img_canny = FindImgContours(img, 300, 500);
        if (isDrawRect)
            detectArmour(draw_img, AimColor, 270);
        return img_canny;
    }
}