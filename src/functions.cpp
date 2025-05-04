#include <iostream>
#include <opencv2/opencv.hpp>
#include "functions.h"
using namespace std;
using namespace cv;

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