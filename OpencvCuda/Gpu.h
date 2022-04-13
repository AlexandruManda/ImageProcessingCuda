#include <opencv2/core/mat.hpp>
using namespace cv;

#ifndef GPU_HPP
#define GPU_HPP

void GPU_TestCanny(Mat img);
void GPU_TestSobel(Mat img);
void GPU_TestLaplacian(Mat img);
void GPU_TestPrewitt(Mat img);

#endif 
