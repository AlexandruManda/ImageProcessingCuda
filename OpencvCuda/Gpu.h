#include <opencv2/core/mat.hpp>
#include "ProjectTypes.h"
using namespace cv;
using namespace cv::cuda;
using namespace std;

#ifndef GPU_HPP
#define GPU_HPP

	void GPU_TestCanny(const Filters& filters, const cv::cuda::HostMem& input, cv::cuda::HostMem& output, Stream stream);
	void GPU_TestSobel(const Filters& filters, const cv::cuda::HostMem& input, cv::cuda::HostMem& output, Stream stream);
	void GPU_TestLaplacian(const Filters& filters, const cv::cuda::HostMem& input, cv::cuda::HostMem& output, Stream stream);
	void GPU_TestPrewitt(const Filters& filters, const cv::cuda::HostMem& input, cv::cuda::HostMem& output, Stream stream);
	void GPU_TestRoberts(const Filters& filters, const cv::cuda::HostMem& input, cv::cuda::HostMem& output, Stream stream);
	void GPU_Initialization(Mat img);
#endif 
