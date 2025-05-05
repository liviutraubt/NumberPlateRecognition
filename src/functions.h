#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

const Mat sharp = (Mat_<float>(3, 3) <<
        0, -1, 0,
       -1, 5, -1,
        0, -1, 0);

const Mat median3 = (Mat_<uchar>(3, 3) <<
        1, 1, 1,
        1, 1, 1,
        1, 1, 1);

typedef struct{
    int min_value;
    int max_value;
} edge_image_values;

bool IsInside(Mat img, int i, int j);
Mat convertToGrayscale(Mat source);
Mat extractBlueMask(Mat source);
Mat convolution(Mat source, Mat kernel);
edge_image_values compute_edge_values(Mat source);
int compute_bimodal_threshold(edge_image_values img_values, int* histogram, float err);
Mat apply_bimodal_thresholding(Mat source, int th);
int* compute_histogram_naive(Mat source);
Mat cannyEdgeDetection(Mat source, int lowThresh, int highThresh);

#endif
