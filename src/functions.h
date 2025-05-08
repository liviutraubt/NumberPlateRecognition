#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

const Mat median3 = (Mat_<uchar>(3, 3) <<
        1, 1, 1,
        1, 1, 1,
        1, 1, 1);

typedef struct{
    int min_value;
    int max_value;
} edge_image_values;

// Kerneluri Sobel
const float Gx[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};
const float Gy[3][3] = {
    {-1, -2, -1},
    { 0,  0,  0},
    { 1,  2,  1}
};


const int dx[8] = { -1, -1,  0, 1, 1,  1,  0, -1 };
const int dy[8] = {  0, -1, -1,-1, 0,  1,  1,  1 };

bool IsInside(Mat img, int i, int j);
Mat convertToGrayscale(Mat source);
Mat extractBlueMask(Mat source);
Mat median_filter(Mat source, Mat kernel);
edge_image_values compute_edge_values(Mat source);
int compute_bimodal_threshold(edge_image_values img_values, int* histogram, float err);
Mat apply_bimodal_thresholding(Mat source, int th);
int* compute_histogram_naive(Mat source);
Mat cannyEdgeDetection(Mat source, int lowThresh, int highThresh);
vector<vector<Point>> extract_all_objects(Mat source);
Rect compute_bounding_box(vector<Point> object);
int compute_area(Rect rect);
Mat extract_license_plate(Mat source, Rect rect);
Mat gaussian_blur_filter(Mat source);

#endif
