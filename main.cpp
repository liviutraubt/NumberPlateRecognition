#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {

    Mat source = imread("D:\\UTCN\\An 3\\Sem 2\\PI\\Proiect\\LicensePlateRecognition\\images\\image.jpeg", -1);

    imshow("Original", source);

    waitKey();
    return 0;
}
