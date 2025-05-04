#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {

    Mat source = imread("images/image.jpeg", -1);

    imshow("Original", source);

    waitKey();
    return 0;
}
