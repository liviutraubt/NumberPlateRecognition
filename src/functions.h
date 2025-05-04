#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat convertToGrayscale(Mat source);
Mat extractBlueMask(Mat source);

#endif
