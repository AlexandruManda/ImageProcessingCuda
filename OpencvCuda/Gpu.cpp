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

void GPU_TestCanny(Mat img) {
	GpuMat imgBlurred;
	namedWindow("GPUCanny", cv::WINDOW_AUTOSIZE);
	int64 t1 = cv::getTickCount();
	cv::cuda::Stream stream[3];

	imgBlurred.upload(img,stream[0]);

	auto gaussianFilter = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, Size(5, 5), 1);
	gaussianFilter->apply(imgBlurred, imgBlurred,stream[1]);

	auto cannyOperator = cv::cuda::createCannyEdgeDetector(50, 100);
	cannyOperator->detect(imgBlurred, imgBlurred, stream[2]);

	imgBlurred.download(img,stream[0]);

	int64 t2 = cv::getTickCount();
	double seconds = (t2 - t1) / cv::getTickFrequency();
	cout << "Canny GPU time: " << seconds << endl;

	imshow("GPUCanny", img);
	waitKey(0);
}

void GPU_TestSobel(Mat img)
{
	int ksize = 3;
	int scale = 1;
	GpuMat imgBlurred, grad_x, grad_y, abs_x, abs_y;
	namedWindow("GPUSobel", cv::WINDOW_FULLSCREEN);
	int64 t0 = cv::getTickCount();


	imgBlurred.upload(img);

	auto gaussianFilter = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, Size(3, 3), 1);
	gaussianFilter->apply(imgBlurred, imgBlurred);

	auto sobelOperator_X = createSobelFilter(CV_8UC1, CV_8UC1, 1, 0, ksize, scale);
	auto sobelOperator_Y = createSobelFilter(CV_8UC1, CV_8UC1, 0, 1, ksize, scale);

	sobelOperator_X->apply(imgBlurred, grad_x);
	sobelOperator_Y->apply(imgBlurred, grad_y);

	cv::cuda::addWeighted(grad_x, 0.5, grad_y, 0.5, 0, imgBlurred);

	imgBlurred.download(img);

	int64 t1 = cv::getTickCount();
	double seconds = (t1 - t0) / cv::getTickFrequency();
	cout << "Sobel GPU time: " << seconds;

	imshow("GPUSobel", img);
	waitKey(0);

}

void GPU_TestLaplacian(Mat img) {
	GpuMat imgBlurred;
	namedWindow("GPULaplacian", cv::WINDOW_FULLSCREEN);

	int64 t0 = cv::getTickCount();

	imgBlurred.upload(img);

	auto gaussianFilter = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, Size(3, 3), 1);
	gaussianFilter->apply(imgBlurred, imgBlurred);

	auto laplacianOperator = cv::cuda::createLaplacianFilter(CV_8UC1, CV_8UC1, 3, 1, BORDER_DEFAULT);
	laplacianOperator->apply(imgBlurred, imgBlurred);

	imgBlurred.download(img);

	int64 t1 = cv::getTickCount();
	double seconds = (t1 - t0) / cv::getTickFrequency();
	cout << "Laplacian GPU time: " << seconds;

	imshow("GPULaplacian", img);
	waitKey(0);
}

void GPU_TestPrewitt(Mat img)
{
	GpuMat imgBlurred,prewitt,prewitt_x,prewitt_y;
	namedWindow("GPUPrewitt", cv::WINDOW_FULLSCREEN);

	int64 t0 = cv::getTickCount();
	
	imgBlurred.upload(img);

	float kernelxData[9] = { 1,1,1,0,0,0,-1,-1,-1 };
	float kernelyData[9] = { -1,0,1,-1,0,1,-1,0,1 };
	GpuMat kernelx = GpuMat(3, 3, CV_32F, kernelxData);
	GpuMat kernely = GpuMat(3, 3, CV_32F, kernelyData);
	
	auto gaussianFilter = createGaussianFilter(CV_8UC1, CV_8UC1, Size(3, 3), 1);
	gaussianFilter->apply(imgBlurred, imgBlurred);

	/*auto convolutionX = createConvolution(Size(3, 3));
	convolutionX->convolve(imgBlurred, kernelx,imgBlurred);*/
	
	/*auto convolutionY = createConvolution(Size(3, 3));
	convolutionY->convolve(imgBlurred, kernely, prewitt_y);*/
	auto linearFilterx = createLinearFilter(CV_8UC1, CV_8UC1, kernelx);
	linearFilterx->apply(imgBlurred, prewitt_x);

	auto linearFiltery = createLinearFilter(CV_8UC1, CV_8UC1, kernely);
	linearFiltery->apply(imgBlurred, prewitt_y);

	cv::cuda::addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0, imgBlurred);

	imgBlurred.download(img);

	int64 t1 = cv::getTickCount();
	double seconds = (t1 - t0) / cv::getTickFrequency();
	cout << "Prewitt GPU time: " << seconds;

	imshow("GPUPrewitt", img);
	waitKey(0);

}
