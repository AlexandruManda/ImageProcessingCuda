#include "Cpu.h"
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

Mat CPU_TestCanny(Mat img) {
	Mat imgBlurred;
	Mat imgCanny;

	int64 t0 = cv::getTickCount();

	GaussianBlur(img, imgBlurred, cv::Size(5, 5), 0.5);
	Canny(imgBlurred, imgCanny, 50, 100);

	int64 t1 = cv::getTickCount();
	double seconds = (t1 - t0) / cv::getTickFrequency();
	cout << "Canny CPU time: " << seconds << endl;
}

Mat CPU_TestSobel(Mat img) {
	Mat imgBlurred;
	Mat grad;
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	int ksize = 3;
	int scale = 1;
	int delta = 1;

	int64 t0 = cv::getTickCount();
	GaussianBlur(img, imgBlurred, cv::Size(5, 5), 0.5);


	Sobel(imgBlurred, grad_x, CV_16S, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
	Sobel(imgBlurred, grad_y, CV_16S, 1, 0, ksize, scale, delta, BORDER_DEFAULT);

	convertScaleAbs(grad_x, abs_grad_x);
	convertScaleAbs(grad_y, abs_grad_y);

	cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

	int64 t1 = cv::getTickCount();
	double seconds = (t1 - t0) / cv::getTickFrequency();
	cout << "Sobel CPU time: " << seconds << endl;
	return grad;
}

Mat CPU_TestLaplacian(Mat img) {

	Mat imgBlurred, dst, abs_dst;
	int64 t0 = cv::getTickCount();
	GaussianBlur(img, imgBlurred, cv::Size(3, 3), 0, 0, BORDER_DEFAULT);

	Laplacian(imgBlurred, dst, CV_16S, 3, 1, 0, BORDER_DEFAULT);

	convertScaleAbs(dst, abs_dst);
	int64 t1 = cv::getTickCount();
	double seconds = (t1 - t0) / cv::getTickFrequency();
	cout << "Laplacian CPU time: " << seconds << endl;

	return abs_dst;
}

Mat CPU_TestPrewitt(Mat img)
{
	Mat imgBlurred,kernelx,kernely;
	Mat prewitt_x,prewitt_y,prewitt;

	int64 t0 = cv::getTickCount();

	float kernelxData[9] = {1,1,1,0,0,0,-1,-1,-1};
	float kernelyData[9] = {-1,0,1,-1,0,1,-1,0,1};
	kernelx = Mat(3, 3, CV_32F, kernelxData);
	kernely = Mat(3, 3, CV_32F, kernelyData);
	GaussianBlur(img, imgBlurred, cv::Size(5, 5), 0.5);

	filter2D(imgBlurred, prewitt_x, -1, kernelx);
	filter2D(imgBlurred, prewitt_y, -1, kernely);
	
	cv::addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0, prewitt);
	int64 t1 = cv::getTickCount();
	double seconds = (t1 - t0) / cv::getTickFrequency();
	cout << "Prewitt CPU time: " << seconds << endl;

	return prewitt;

}

Mat CPU_TestRoberts(Mat img)
{
	Mat imgBlurred, kernelx, kernely;
	Mat robert_x, robert_y, robert;

	int64 t0 = cv::getTickCount();

	float kernelxData[4] = {1,0,0,-1};
	float kernelyData[4] = { 0,1,-1,0 };
	kernelx = Mat(2, 2, CV_32F, kernelxData);
	kernely = Mat(2, 2, CV_32F, kernelyData);
	GaussianBlur(img, imgBlurred, cv::Size(3, 3), 0.5);

	filter2D(imgBlurred, robert_x, -1, kernelx);
	filter2D(imgBlurred, robert_y, -1, kernely);
	
	cv::addWeighted(robert_x, 2, robert_y, 2, 0, robert);
	int64 t1 = cv::getTickCount();
	double seconds = (t1 - t0) / cv::getTickFrequency();
	cout << "Roberts CPU time: " << seconds << endl;

	return robert;

}
