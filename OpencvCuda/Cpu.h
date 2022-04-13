#include <opencv2/core/mat.hpp>
using namespace cv;

#ifndef CPU_HPP
#define CPU_HPP

void CPU_TestCanny(Mat img);
void CPU_TestSobel(Mat img);
void CPU_TestLaplacian(Mat img);
void CPU_TestPrewitt(Mat img);
void CPU_TestRoberts(Mat img);

#endif 
