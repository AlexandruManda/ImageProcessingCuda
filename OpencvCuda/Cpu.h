#include <opencv2/core/mat.hpp>
using namespace cv;

#ifndef CPU_HPP
#define CPU_HPP

Mat CPU_TestCanny(Mat img);
Mat CPU_TestSobel(Mat img);
Mat CPU_TestLaplacian(Mat img);
Mat CPU_TestPrewitt(Mat img);
Mat CPU_TestRoberts(Mat img);

#endif 
