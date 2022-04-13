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
#include "Gpu.h" 
#include "Cpu.h"

#define IMG_PATH "images/895.jpg"

using namespace std;
using namespace cv;
using namespace cv::cuda;

void WEBCAM_Test();

int main()
{
	Mat img = cv::imread(IMG_PATH, 0);

	//printCudaDeviceInfo(0);
	//CPU_TestCanny(img);
	//CPU_TestSobel(img);
	CPU_TestLaplacian(img);
	//GPU_TestCanny(img);
	//GPU_TestLaplacian(img);
	//WEBCAM_Test();
	//GPU_TestPrewitt(img);
	//CPU_TestPrewitt(img);
	//CPU_TestRoberts(img);
	//GPU_TestSobel(img);
}



void WEBCAM_Test()
{
	VideoCapture cap(0);
	Mat img, imgBlurred;
	GpuMat imgGpu, mat, grad_x, grad_y;
	vector<GpuMat> gpuMats;
	int ksize = 3;
	int scale = 1;
	while (cap.isOpened()) {
		auto start = getTickCount();

		cap.read(img);

		imgGpu.upload(img);

		cv::cuda::cvtColor(imgGpu, imgGpu, COLOR_BGR2GRAY);
		auto gaussianFilter = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, Size(3, 3), 0.5);
		gaussianFilter->apply(imgGpu, imgGpu);

		//Canny filter to each frame from webcam
		auto cannyOperator = cv::cuda::createCannyEdgeDetector(20, 50);
		cannyOperator->detect(imgGpu, imgGpu);

		//Laplacian filter to each frame from webcam
		/*gaussianFilter->apply(imgGpu, imgGpu);
		auto laplacianOperator = cv::cuda::createLaplacianFilter(CV_8UC1, CV_8UC1,3, 2, BORDER_DEFAULT);
		laplacianOperator->apply(imgGpu, imgGpu);*/

		//Sobel filter to each frame from webcam
		auto sobelOperator_X = createSobelFilter(CV_8UC1, CV_8UC1, 1, 0, ksize, scale);
		auto sobelOperator_Y = createSobelFilter(CV_8UC1, CV_8UC1, 0, 1, ksize, scale);
		sobelOperator_X->apply(imgGpu, grad_x);
		sobelOperator_Y->apply(imgGpu, grad_y);
		cv::cuda::addWeighted(grad_x, 0.5, grad_y, 0.5, 0, imgGpu);

		imgGpu.download(img);


		auto end = getTickCount();
		auto totalTime = (end - start) / getTickFrequency();
		auto fps = 1 / totalTime;
		putText(img, "FPS: " + to_string(int(fps)), Point(50, 50), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 200, 10));
		imshow("Image", img);
		if (waitKey(1) == 'q') {
			break;
		}

	}

	cap.release();
	destroyAllWindows();

}
