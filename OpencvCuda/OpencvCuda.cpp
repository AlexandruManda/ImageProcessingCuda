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
#include "omp.h"
#include "Cpu.h"
#include <mutex>

#define IMG_PATH "data/*.png"

using namespace std;
using namespace cv;
using namespace cv::cuda;

void WEBCAM_Test();

vector<String> fileNames;
float kernelxData[9] = { 1,1,1,0,0,0,-1,-1,-1 };
float kernelyData[9] = { -1,0,1,-1,0,1,-1,0,1 };
float robertsKernelxData[4] = { 1,0,0,-1 };
float robertsKernelyData[4] = { 0,1,-1,0 };
Mat kernelxPrewitt = Mat(3, 3, CV_32F, kernelxData);
Mat kernelyPrewitt = Mat(3, 3, CV_32F, kernelyData);
Mat kernelxRoberts = Mat(2, 2, CV_32F, robertsKernelxData);
Mat kernelyRoberts = Mat(2, 2, CV_32F, robertsKernelyData);

Filters filters = {
	createGaussianFilter(CV_8UC1, CV_8UC1, Size(3, 3), 0.5),
	createSobelFilter(CV_8UC1, CV_8UC1, 1, 0, 3, 1),
	createSobelFilter(CV_8UC1, CV_8UC1, 0, 1, 3, 1),
	createLaplacianFilter(CV_8UC1, CV_8UC1, 3, 1, BORDER_DEFAULT),
	createLinearFilter(CV_8UC1, CV_8UC1, kernelxPrewitt),
	createLinearFilter(CV_8UC1, CV_8UC1, kernelyPrewitt),
	createLinearFilter(CV_8UC1, CV_8UC1, kernelxRoberts),
	createLinearFilter(CV_8UC1, CV_8UC1, kernelyRoberts),
	createCannyEdgeDetector(50, 100)
};

int main()
{
	//Add filename to vector of strings
	glob(IMG_PATH, fileNames, false);
	
	//Get size of the filenameVector
	size_t  count = fileNames.size(); 

/*------------Initialization of the GPU ------------*/

	Stream stream2;
	Mat img = imread("images/895.jpg", IMREAD_GRAYSCALE);
	GPU_Initialization(img);
	/*HostMem init_input (img, HostMem::AllocType::PAGE_LOCKED);
	HostMem init_output(HostMem::AllocType::PAGE_LOCKED);
	GPU_TestCanny(filters, init_input, init_output, stream2);*/
/*-------------End of initialization --------------*/
	Stream stream[8];
	int64 t0 = cv::getTickCount();	
	
#pragma omp parallel for num_threads(5)
	for (int i = 0; i < count; i++) {
		Mat img2 = imread(fileNames[i],IMREAD_GRAYSCALE);
		
		HostMem pinned_input(img2,HostMem::AllocType::PAGE_LOCKED);
		HostMem pinned_output(HostMem::AllocType::PAGE_LOCKED);
		GPU_TestSobel(filters,pinned_input,pinned_output,stream[omp_get_thread_num()]);
		/*Mat pinned_output = CPU_TestRoberts(img2);*/
	

		stringstream ss;
		ss << "Processed/image" << (i) << ".png";
		string filename = ss.str();
		imwrite(filename, pinned_output);
			
	}

	int64 t1 = cv::getTickCount();
	double seconds = (t1 - t0) / cv::getTickFrequency();
	std::cout << seconds;

	//Mat img2 = imread("images/895.jpg", IMREAD_GRAYSCALE);
	//HostMem pinned_input(img2, HostMem::AllocType::PAGE_LOCKED);
	//HostMem pinned_output(HostMem::AllocType::PAGE_LOCKED);
	//GPU_TestPrewitt(pinned_input, pinned_output);
	//imwrite("Processed/lenny2.jpg",pinned_output);

	//printCudaDeviceInfo(0);
	/*CPU_TestCanny(img);
	GPU_TestCanny(img);*/
	/*CPU_TestSobel(img);
	GPU_TestSobel(img);*/
	/*CPU_TestLaplacian(img);
	GPU_TestLaplacian(img);*/
	
	/*CPU_TestPrewitt(img);
	GPU_TestPrewitt(img);*/

	/*CPU_TestRoberts(img);
	GPU_TestRoberts(img);*/
	
	//WEBCAM_Test();
	return 0;
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
		cv::cvtColor(img, img, COLOR_BGR2GRAY);
		HostMem pinned_input(img, HostMem::AllocType::PAGE_LOCKED);
		HostMem pinned_output(HostMem::AllocType::PAGE_LOCKED);
		//GPU_TestSobel(filters,pinned_input, pinned_output);
		auto end = getTickCount();
		auto totalTime = (end - start) / getTickFrequency();
		auto fps = 1 / totalTime;
		putText(pinned_output, "FPS: " + to_string(int(fps)), Point(50, 50), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 200, 10));
		imshow("Image", pinned_output);
		if (waitKey(1) == 'q') {
			break;
		}

	}

	cap.release();
	destroyAllWindows();

}
