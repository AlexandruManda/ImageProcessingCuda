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

#define IMG_PATH "images/*.jpg"

using namespace std;
using namespace cv;
using namespace cv::cuda;

void WEBCAM_Test();

vector<String> fileNames;

int main()
{
	Mutex mutex;
	//Add filename to vector of strings
	glob(IMG_PATH, fileNames, false);
	
	//Get size of the filenameVector
	int  count = fileNames.size(); 

/*------------Initialization of the GPU ------------*/

	Mat img = imread("images/895.jpg", IMREAD_GRAYSCALE);
	GPU_Initialization(img);

/*-------------End of initialization --------------*/
	
	int64 t0 = cv::getTickCount();

#pragma omp parallel for num_threads(12)
	for (int i = 0; i < count; i++) {
		Mat img2 = imread(fileNames[i], IMREAD_GRAYSCALE);
		
		HostMem pinned_input(img2,HostMem::AllocType::PAGE_LOCKED);
		HostMem pinned_output(HostMem::AllocType::PAGE_LOCKED);
		GPU_TestCanny(img2,pinned_input,pinned_output);

		//Mat pinned_output = CPU_TestCanny(img2);
		stringstream ss;
		ss << "Processed/image" << (i) << ".jpg";
		string filename = ss.str();
		imwrite(filename, pinned_output);
			
	}
	int64 t1 = cv::getTickCount();
	double seconds = (t1 - t0) / cv::getTickFrequency();
	cout << seconds;

	
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
		//img = GPU_TestPrewitt(img);

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
