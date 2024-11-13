#include<iostream>
#include <thread>
#include <boost/filesystem.hpp>
#include<string>
#include <fstream>
#include<json.hpp>
#include <opencv2/opencv.hpp>


using namespace cv;
namespace ccustomutils{
    // Function to convert a Base64 string to cv::Mat image
    cv::Mat base64ToMat(const std::string base64_data);
    bool doesFileExist(const std::string& filepath);
    float* blobFromImage(Mat& img);
    std::string millisecondsToDateTimeString();
};