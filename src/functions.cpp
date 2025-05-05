#include <iostream>
#include <opencv2/opencv.hpp>
#include "functions.h"
using namespace std;
using namespace cv;

bool IsInside(Mat img, int i, int j){
    int rows, cols;
    rows = img.rows;
    cols = img.cols;

    if(i >= 0 && i < rows && j >= 0 && j < cols)
        return true;

    return false;
}

Mat convertToGrayscale(Mat source) {
    int rows = source.rows;
    int cols = source.cols;
    int converted_value;
    Mat grayscale_image = Mat(rows, cols, CV_8UC1);

    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++) {
            converted_value = (source.at<Vec3b >(i,j)[0] + source.at<Vec3b >(i,j)[1] + source.at<Vec3b >(i,j)[2]) / 3;
            grayscale_image.at<uchar>(i, j) = converted_value;
        }
    }

    return grayscale_image;
}

Mat extractBlueMask(Mat source){
    int rows = source.rows, cols = source.cols;
    Mat BlueMask = Mat(rows, cols, CV_8UC1, 255);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Vec3b pixel = source.at<Vec3b>(i, j);
            if (pixel[0] <= pixel[1] + 30 || pixel[0] <= pixel[2] + 30) {
                BlueMask.at<uchar>(i, j) = 0;
            }
        }
    }

    return BlueMask;
}

Mat convolution(Mat source, Mat kernel){
    int rows = source.rows, cols = source.cols;
    int kernel_rows = kernel.rows, kernel_cols = kernel.cols;
    int kernel_center_x = kernel_cols / 2;
    int kernel_center_y = kernel_rows / 2;
    Mat result = Mat(rows, cols, CV_8UC1, Scalar(0));

    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            vector<uchar> neighborhood;

            for (int m = -kernel_center_y; m <= kernel_center_y; m++) {
                for (int n = -kernel_center_x; n <= kernel_center_x; n++) {
                    int y = i + m;
                    int x = j + n;

                    if (IsInside(source, y, x)) {
                        neighborhood.push_back(source.at<uchar>(y, x));
                    }
                }
            }

            for (size_t i = 0; i < neighborhood.size(); i++) {
                for (size_t j = 0; j < neighborhood.size() - i - 1; j++) {
                    if (neighborhood[j] > neighborhood[j + 1]) {
                        swap(neighborhood[j], neighborhood[j + 1]);
                    }
                }
            }
            result.at<uchar>(i, j) = neighborhood[neighborhood.size() / 2];
        }
    }

    return result;
}

int* compute_histogram_naive(Mat source){
    int* histogram = (int*)calloc(256, sizeof(int));

    int rows = source.rows, cols = source.cols;
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            histogram[source.at<uchar>(i, j)]++;
        }
    }

    return histogram;
}

edge_image_values compute_edge_values(Mat source){
    int min = 255, max = 0;

    int rows = source.rows, cols = source.cols;
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            int pixel_value = source.at<uchar>(i, j);
            if(pixel_value < min)
                min = pixel_value;
            if(pixel_value > max)
                max = pixel_value;
        }
    }

    return {min, max};
}

int compute_bimodal_threshold(edge_image_values img_values, int* histogram, float err){
    int min, max;
    float Tk, Tk_1;
    float mean_1, mean_2, no_pixels_mean_1, no_pixels_mean_2;

    min = img_values.min_value;
    max = img_values.max_value;

    Tk = (int)(min + max) /2;

    do{
        Tk_1 = Tk;

        mean_1 = 0; no_pixels_mean_1 = 0;
        mean_2 = 0; no_pixels_mean_2 = 0;

        for(int i = min; i <= Tk; i++){
            mean_1 += histogram[i] * i;
            no_pixels_mean_1 += histogram[i];
        }

        for(int i = Tk + 1; i <= max; i++){
            mean_2 += histogram[i] * i;
            no_pixels_mean_2 += histogram[i];
        }

        if(no_pixels_mean_1 != 0)
            mean_1 /= no_pixels_mean_1;

        if(no_pixels_mean_2 != 0)
            mean_2 /= no_pixels_mean_2;

        Tk = (int)(mean_1 + mean_2) / 2;

    }while(abs(Tk - Tk_1) > err);

    return Tk;
}

Mat apply_bimodal_thresholding(Mat source, int th){
    Mat dst;

    dst = Mat(source.rows, source.cols, CV_8UC1, 255);

    for(int i = 0; i < source.rows; i++){
        for(int j = 0; j < source.cols; j++){
            if(source.at<uchar>(i, j) <= th){
                dst.at<uchar>(i, j) = 0;
            }
        }
    }

    return dst;
}

Mat cannyEdgeDetection(Mat source, int lowThresh, int highThresh){

}