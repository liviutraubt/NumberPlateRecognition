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
            if (pixel[0] <= pixel[1] + 50 || pixel[0] <= pixel[2] + 50) {
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
    int rows = source.rows, cols = source.cols;
    Mat gradient = Mat(rows, cols, CV_32FC1, Scalar(0));
    Mat direction = Mat(rows, cols, CV_32FC1, Scalar(0));
    Mat edges = Mat(rows, cols, CV_8UC1, Scalar(0));

    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            float sumX = 0, sumY = 0;

            for (int m = -1; m <= 1; m++) {
                for (int n = -1; n <= 1; n++) {
                    uchar pixel = source.at<uchar>(i + m, j + n);
                    sumX += pixel * Gx[m + 1][n + 1];
                    sumY += pixel * Gy[m + 1][n + 1];
                }
            }

            float mag = sqrt(sumX * sumX + sumY * sumY);
            float angle = atan2(sumY, sumX);

            gradient.at<float>(i, j) = mag;
            direction.at<float>(i, j) = angle;
        }
    }

    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            float mag = gradient.at<float>(i, j);

            if (mag >= highThresh) {
                edges.at<uchar>(i, j) = 255;
            } else if (mag >= lowThresh) {
                bool connected = false;
                for (int m = -1; m <= 1; m++) {
                    for (int n = -1; n <= 1; n++) {
                        if (gradient.at<float>(i + m, j + n) >= highThresh) {
                            connected = true;
                        }
                    }
                }
                if (connected)
                    edges.at<uchar>(i, j) = 255;
            }
        }
    }

    return edges;
}

vector<vector<Point>> extract_all_contours_from_edges(Mat source) {
    int rows = source.rows;
    int cols = source.cols;
    Mat visited = Mat(rows, cols, CV_8UC1, Scalar(0));
    vector<vector<Point>> contours;

    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            if (source.at<uchar>(y, x) == 0 && visited.at<uchar>(y, x) == 0) {
                vector<Point> contur;
                queue<Point> q;
                q.push(Point(x, y));
                visited.at<uchar>(y, x) = 1;

                while (!q.empty()) {
                    Point p = q.front();
                    q.pop();
                    contur.push_back(p);

                    for (int k = 0; k < 8; ++k) {
                        int nx = p.x + dx[k];
                        int ny = p.y + dy[k];

                        if (IsInside(source, ny, nx) && source.at<uchar>(ny, nx) == 0 && visited.at<uchar>(ny, nx) == 0) {
                            q.push(Point(nx, ny));
                            visited.at<uchar>(ny, nx) = 1;
                        }
                    }
                }

                if (contur.size() >= 100) {
                    contours.push_back(contur);
                }
            }
        }
    }

    return contours;
}
