#include "Gpu.h"
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafeatures2d.hpp>

using namespace std;
using namespace cv;
using namespace cv::cuda;


void GPU_TestCanny(const Filters& filters,const cv::cuda::HostMem& input, cv::cuda::HostMem& output, Stream stream) {
	GpuMat imgBlurred;
	

	imgBlurred.upload(input,stream);

	filters.gaussian->apply(imgBlurred, imgBlurred,stream);
	//stream.waitForCompletion();
	auto cannyOperator = createCannyEdgeDetector(50, 100);
	cannyOperator->detect(imgBlurred, imgBlurred, stream);
	//filters.canny->detect(imgBlurred, imgBlurred,stream);
	//stream.waitForCompletion();

	imgBlurred.download(output,stream);
	//stream.waitForCompletion();
	
}

void GPU_TestSobel(const Filters& filters,const cv::cuda::HostMem& input, cv::cuda::HostMem& output, Stream stream)
{

	GpuMat imgBlurred, grad_x, grad_y, abs_x, abs_y;


	imgBlurred.upload(input,stream);


	filters.gaussian->apply(imgBlurred, imgBlurred, stream);
	

	filters.sobelx->apply(imgBlurred, grad_x, stream);
	filters.sobely->apply(imgBlurred, grad_y, stream);
	

	cv::cuda::addWeighted(grad_x, 1, grad_y, 1, 0, imgBlurred,-1,stream);
	

	imgBlurred.download(output,stream);
	
}

void GPU_TestLaplacian(const Filters& filters, const cv::cuda::HostMem& input, cv::cuda::HostMem& output, Stream stream) {
	GpuMat imgBlurred;


	imgBlurred.upload(input,stream);


	filters.gaussian->apply(imgBlurred, imgBlurred,stream);


	filters.laplacian->apply(imgBlurred, imgBlurred,stream);

	imgBlurred.download(output,stream);

}

void GPU_TestPrewitt(const Filters& filters, const cv::cuda::HostMem& input, cv::cuda::HostMem& output, Stream stream)
{
	GpuMat imgBlurred,prewitt,prewitt_x,prewitt_y;

	
	imgBlurred.upload(input,stream);

	
	filters.gaussian->apply(imgBlurred, imgBlurred,stream);
	
	filters.prewittx->apply(imgBlurred, prewitt_x,stream);

	filters.prewitty->apply(imgBlurred, prewitt_y,stream);

	cv::cuda::addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0, imgBlurred,-1,stream);

	imgBlurred.download(output,stream);

	
}

void GPU_TestRoberts(const Filters& filters, const cv::cuda::HostMem& input, cv::cuda::HostMem& output, Stream stream)
{
	GpuMat imgBlurred, robert, robert_x, robert_y;


	imgBlurred.upload(input,stream);

	filters.gaussian->apply(imgBlurred, imgBlurred,stream);
	
	filters.robertx->apply(imgBlurred, robert_x,stream);
	
	filters.roberty->apply(imgBlurred, robert_x,stream);
	
	cv::cuda::addWeighted(robert_x, 2, robert_x, 2, 0, imgBlurred,-1,stream);
	
	imgBlurred.download(output,stream);


}


void GPU_Initialization(Mat img)
{
	GpuMat imgBlurred, prewitt, prewitt_x, prewitt_y;

	imgBlurred.upload(img);

	float kernelxData[4] = { 1,0,0,-1 };
	float kernelyData[4] = { 0,1,-1,0 };
	Mat kernelx = Mat(2, 2, CV_32F, kernelxData);
	Mat kernely = Mat(2, 2, CV_32F, kernelyData);

	auto gaussianFilter = createGaussianFilter(CV_8UC1, CV_8UC1, Size(3, 3), 1);
	gaussianFilter->apply(imgBlurred, imgBlurred);

	auto linearFilterx = createLinearFilter(CV_8UC1, CV_8UC1, kernelx);
	linearFilterx->apply(imgBlurred, prewitt_x);

	auto linearFiltery = createLinearFilter(CV_8UC1, CV_8UC1, kernely);
	linearFiltery->apply(imgBlurred, prewitt_y);

	cv::cuda::addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0, imgBlurred);
	imgBlurred.download(img);

}


