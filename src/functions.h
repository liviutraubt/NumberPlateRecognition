#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

const Mat kernel = (Mat_<float>(3, 3) <<
        0, -1, 0,
       -1, 5, -1,
        0, -1, 0);

bool IsInside(Mat img, int i, int j);
Mat convertToGrayscale(Mat source);
Mat extractBlueMask(Mat source);
Mat convolution(Mat source, Mat kernel);

#endif
