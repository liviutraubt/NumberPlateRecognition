#include <iostream>
#include "src/functions.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {

    // 1. Citim imaginea
    Mat source = imread("images/test.jpeg");

    // 2. Convertim în grayscale și păstrăm o mască albastră pentru bara din stânga
    Mat gray = convertToGrayscale(source), blue_mask = extractBlueMask(source);
    // vector<Mat> channels;
    // split(source, channels);  // B, G, R
    //
    // // mască: albastru semnificativ mai intens decât celelalte
    // Mat condition1 = channels[0] > channels[1] + 30;
    // Mat condition2 = channels[0] > channels[2] + 30;
    // bitwise_and(condition1, condition2, blue_mask);
    //
    // cvtColor(source, gray, COLOR_BGR2GRAY);



    // 3. Aplicăm sharpening
    Mat sharpened;
    Mat kernel = (Mat_<float>(3, 3) <<
        0, -1, 0,
       -1, 5, -1,
        0, -1, 0);
    filter2D(gray, sharpened, CV_8U, kernel);

    // 4. Aplicăm filtru median
    Mat medianed;
    medianBlur(sharpened, medianed, 3);

    // 5. Binarizare (threshold automatizat)
    Mat binary;
    double thresh_val = mean(medianed)[0];
    threshold(medianed, binary, thresh_val, 255, THRESH_BINARY);

    // 6. Detecție margini (Canny)
    Mat edges;
    int lowThresh = 50, highThresh = 150;
    Canny(binary, edges, lowThresh, highThresh);

    // 7. Contururi
    vector<vector<Point>> contours;
    findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 8. Detectare plăcuțe după aspect ratio
    Mat result = source.clone();
    for (const auto& contour : contours) {
        Rect rect = boundingRect(contour);
        double aspect = (double)rect.width / rect.height;

        if (aspect > 4.0 && aspect < 5.2 && rect.area() > 3000) {

            // Define a region to the left of the rectangle
            Rect leftRegion(max(0, rect.x - rect.width), rect.y, rect.width, rect.height);

            // Check if there are any non-zero pixels in the blue_mask within the left region
            Mat leftROI = blue_mask(leftRegion);
            if (countNonZero(leftROI) > 0) {
                rectangle(result, rect, Scalar(0, 255, 0), 2);
            }
        }
    }

    //output de test
    // imwrite("images/output/result.jpg", result);

    // Afișare rezultate intermediare
    namedWindow("Original", WINDOW_NORMAL);
    resizeWindow("Original", 600, 400);
    imshow("Original", source);

    namedWindow("Blue Mask", WINDOW_NORMAL);
    resizeWindow("Blue Mask", 600, 400);
    imshow("Blue Mask", blue_mask);

    namedWindow("Sharpened", WINDOW_NORMAL);
    resizeWindow("Sharpened", 600, 400);
    imshow("Sharpened", sharpened);

    namedWindow("Medianed", WINDOW_NORMAL);
    resizeWindow("Medianed", 600, 400);
    imshow("Medianed", medianed);

    namedWindow("Binary", WINDOW_NORMAL);
    resizeWindow("Binary", 600, 400);
    imshow("Binary", binary);

    namedWindow("Edges", WINDOW_NORMAL);
    resizeWindow("Edges", 600, 400);
    imshow("Edges", edges);

    namedWindow("Detected Plates", WINDOW_NORMAL);
    resizeWindow("Detected Plates", 600, 400);
    imshow("Detected Plates", result);

    waitKey(0);
    return 0;
}
