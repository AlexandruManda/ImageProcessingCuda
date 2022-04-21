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

void GPU_TestCanny(Mat img, const cv::cuda::HostMem& input, cv::cuda::HostMem& output) {
	GpuMat imgBlurred;
	Stream stream1, stream2, stream3;

	imgBlurred.upload(input,stream1);
	stream1.waitForCompletion();

	auto gaussianFilter = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, Size(5, 5), 0.5);
	gaussianFilter->apply(imgBlurred, imgBlurred,stream2);
	stream2.waitForCompletion();

	auto cannyOperator = cv::cuda::createCannyEdgeDetector(50, 100);
	cannyOperator->detect(imgBlurred, imgBlurred,stream3);
	//stream3.waitForCompletion();

	imgBlurred.download(output,stream1);

	
}

void GPU_TestSobel(Mat img, const cv::cuda::HostMem& input, cv::cuda::HostMem& output)
{
	int ksize = 3;
	int scale = 1;
	GpuMat imgBlurred, grad_x, grad_y, abs_x, abs_y;
	Stream stream1, stream2, stream3, stream4;


	imgBlurred.upload(input,stream1);
	stream1.waitForCompletion();

	auto gaussianFilter = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, Size(3, 3), 1);
	gaussianFilter->apply(imgBlurred, imgBlurred,stream2);
	stream2.waitForCompletion();

	auto sobelOperator_X = createSobelFilter(CV_8UC1, CV_8UC1, 1, 0, ksize, scale);
	auto sobelOperator_Y = createSobelFilter(CV_8UC1, CV_8UC1, 0, 1, ksize, scale);

	sobelOperator_X->apply(imgBlurred, grad_x,stream3);
	stream3.waitForCompletion();

	sobelOperator_Y->apply(imgBlurred, grad_y,stream4);
	stream4.waitForCompletion();

	cv::cuda::addWeighted(grad_x, 1, grad_y, 1, 0, imgBlurred,-1,stream4);
	stream4.waitForCompletion();

	imgBlurred.download(output,stream1);

	
}

void GPU_TestLaplacian(Mat img, const cv::cuda::HostMem& input, cv::cuda::HostMem& output) {
	GpuMat imgBlurred;
	Stream stream1, stream2, stream3;


	imgBlurred.upload(input,stream1);
	stream1.waitForCompletion();

	auto gaussianFilter = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, Size(3, 3), 1);
	gaussianFilter->apply(imgBlurred, imgBlurred,stream2);
	stream2.waitForCompletion();
	
	auto laplacianOperator = cv::cuda::createLaplacianFilter(CV_8UC1, CV_8UC1, 3, 1, BORDER_DEFAULT);
	laplacianOperator->apply(imgBlurred, imgBlurred,stream3);
	stream3.waitForCompletion();

	imgBlurred.download(output,stream1);

	
}

void GPU_TestPrewitt(Mat img, const cv::cuda::HostMem& input, cv::cuda::HostMem& output)
{
	GpuMat imgBlurred,prewitt,prewitt_x,prewitt_y;
	Stream stream1, stream2, stream3, stream4;
	
	imgBlurred.upload(input,stream1);

	stream1.waitForCompletion();

	float kernelxData[9] = { 1,1,1,0,0,0,-1,-1,-1 };
	float kernelyData[9] = { -1,0,1,-1,0,1,-1,0,1 };
	Mat kernelx = Mat(3, 3, CV_32F, kernelxData);
	Mat kernely = Mat(3, 3, CV_32F, kernelyData);
	
	auto gaussianFilter = createGaussianFilter(CV_8UC1, CV_8UC1, Size(3, 3), 1);
	gaussianFilter->apply(imgBlurred, imgBlurred,stream1);
	stream1.waitForCompletion();
	
	auto linearFilterx = createLinearFilter(CV_8UC1, CV_8UC1, kernelx);
	linearFilterx->apply(imgBlurred, prewitt_x,stream2);
	stream2.waitForCompletion();

	auto linearFiltery = createLinearFilter(CV_8UC1, CV_8UC1, kernely);
	linearFiltery->apply(imgBlurred, prewitt_y,stream3);
	stream3.waitForCompletion();

	cv::cuda::addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0, imgBlurred,-1,stream4);
	stream4.waitForCompletion();

	imgBlurred.download(output,stream1);

	
}

void GPU_TestRoberts(Mat img, const cv::cuda::HostMem& input, cv::cuda::HostMem& output)
{
	GpuMat imgBlurred, robert, robert_x, robert_y;
	Stream stream1, stream2,stream3,stream4;

	imgBlurred.upload(input,stream1);

	stream1.waitForCompletion();

	float kernelxData[4] = { 1,0,0,-1 };
	float kernelyData[4] = { 0,1,-1,0 };
	Mat kernelx = Mat(2, 2, CV_32F, kernelxData);
	Mat kernely = Mat(2, 2, CV_32F, kernelyData);

	auto gaussianFilter = createGaussianFilter(CV_8UC1, CV_8UC1, Size(3, 3), 0.5);
	gaussianFilter->apply(imgBlurred, imgBlurred,stream1);
	stream1.waitForCompletion();
	
	auto linearFilterx = createLinearFilter(CV_8UC1, CV_8UC1, kernelx);
	linearFilterx->apply(imgBlurred, robert_x,stream2);
	stream2.waitForCompletion();
	
	auto linearFiltery = createLinearFilter(CV_8UC1, CV_8UC1, kernely);
	linearFiltery->apply(imgBlurred, robert_x,stream3);
	stream3.waitForCompletion();
	
	cv::cuda::addWeighted(robert_x, 2, robert_x, 2, 0, imgBlurred,-1,stream4);
	stream4.waitForCompletion();
	
	imgBlurred.download(output,stream1);


}

Mat GPU_TestRoberts(Mat img)
{
	GpuMat imgBlurred, robert, robert_x, robert_y;
	Stream stream1, stream2, stream3, stream4;

	imgBlurred.upload(img, stream1);

	stream1.waitForCompletion();

	float kernelxData[4] = { 1,0,0,-1 };
	float kernelyData[4] = { 0,1,-1,0 };
	Mat kernelx = Mat(2, 2, CV_32F, kernelxData);
	Mat kernely = Mat(2, 2, CV_32F, kernelyData);

	auto gaussianFilter = createGaussianFilter(CV_8UC1, CV_8UC1, Size(3, 3), 0.5);
	gaussianFilter->apply(imgBlurred, imgBlurred, stream1);
	stream1.waitForCompletion();

	auto linearFilterx = createLinearFilter(CV_8UC1, CV_8UC1, kernelx);
	linearFilterx->apply(imgBlurred, robert_x, stream2);
	stream2.waitForCompletion();

	auto linearFiltery = createLinearFilter(CV_8UC1, CV_8UC1, kernely);
	linearFiltery->apply(imgBlurred, robert_x, stream3);
	stream3.waitForCompletion();

	cv::cuda::addWeighted(robert_x, 2, robert_x, 2, 0, imgBlurred, -1, stream4);
	stream4.waitForCompletion();

	imgBlurred.download(img, stream1);

	return img;

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


