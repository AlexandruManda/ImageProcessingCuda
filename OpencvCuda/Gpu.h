#include <opencv2/core/mat.hpp>
using namespace cv;
using namespace cv::cuda;

#ifndef GPU_HPP
#define GPU_HPP

void GPU_TestCanny(Mat img, const cv::cuda::HostMem& input, cv::cuda::HostMem& output);
void GPU_TestSobel(Mat img, const cv::cuda::HostMem& input, cv::cuda::HostMem& output);
void GPU_TestLaplacian(Mat img, const cv::cuda::HostMem& input, cv::cuda::HostMem& output);
void GPU_TestPrewitt(Mat img, const cv::cuda::HostMem& input, cv::cuda::HostMem& output);
void GPU_TestRoberts(Mat img, const cv::cuda::HostMem& input, cv::cuda::HostMem& output);
Mat GPU_TestRoberts(Mat img);
void GPU_Initialization(Mat img);
#endif 
