#ifndef TESTING_H
#define TESTING_H

#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

void draw_ground_truth(const string& imagePath, const string& outputFilePath);
vector<Rect> read_ground_truth(const string& filePath);
double compute_IoU(const Rect& a, const Rect& b);
void evaluate_detections(const vector<Rect>& detections, const vector<Rect>& groundTruths);

#endif