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