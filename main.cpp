#include <iostream>
#include "src/functions.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {

    // 1. Citim imaginea
    Mat source = imread("images/image.jpeg");

    // 2. Convertim în grayscale și păstrăm o mască albastră pentru bara din stânga
    Mat gray = convertToGrayscale(source), blue_mask = extractBlueMask(source);

    // 3. Aplicăm sharpening
    Mat sharpened = convolution(gray, sharp);

    // 4. Aplicăm filtru median
    Mat medianed = convolution(sharpened, median3);

    // 5. Binarizare (threshold automatizat)
    edge_image_values values_img = compute_edge_values(medianed);
    int* histogram = compute_histogram_naive(medianed);
    int th = compute_bimodal_threshold(values_img, histogram, 0.1);
    Mat binary = apply_bimodal_thresholding(medianed, th);

    // 6. Detecție margini (Canny)
    Mat edges = cannyEdgeDetection(binary, 50, 150);

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
                rectangle(result, rect, Scalar(0, 255, 0), 4);
            }
        }
    }

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
