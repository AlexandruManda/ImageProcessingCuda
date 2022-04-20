#include <opencv2/core/mat.hpp>
using namespace cv;

#ifndef GPU_HPP
#define GPU_HPP

Mat GPU_TestCanny(Mat img);
Mat GPU_TestSobel(Mat img);
Mat GPU_TestLaplacian(Mat img);
Mat GPU_TestPrewitt(Mat img);
Mat GPU_TestRoberts(Mat img);
void GPU_Initialization(Mat img);
#endif 
