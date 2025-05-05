#include <iostream>
#include "src/functions.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {

    // 1. Citim imaginea
    Mat source = imread("images/image.jpeg");

    // 2. Convertim in grayscale si pastram o masca albastra pentru bara din stanga
    Mat gray = convertToGrayscale(source), blue_mask = extractBlueMask(source);

    // 3. Aplicam sharpening
    Mat sharpened = convolution(gray, sharp);

    // 4. Aplicam filtru median
    Mat medianed = convolution(sharpened, median3);

    // 5. Binarizare (threshold automatizat)
    edge_image_values values_img = compute_edge_values(medianed);
    int* histogram = compute_histogram_naive(medianed);
    int th = compute_bimodal_threshold(values_img, histogram, 0.1);
    Mat binary = apply_bimodal_thresholding(medianed, th);

    // 6. Detectie margini (Canny)
    Mat edges = cannyEdgeDetection(binary, 50, 150);

    // 7. Contururi
    vector<vector<Point>> contours = extract_all_contours_from_edges(edges);

    // 8. Detectare placuțe dupa aspect ratio si dimensiune
    Mat result = source.clone();
    for (const auto& contour : contours) {
        Rect rect = boundingRect(contour);
        double aspect = (double)rect.width / rect.height;

        if (aspect > 4.0 && aspect < 5.2 && rect.area() > 3000) {
            Rect leftRegion(max(0, rect.x - rect.width), rect.y, rect.width, rect.height);

            if (leftRegion.x >= 0 && leftRegion.y >= 0 &&
                leftRegion.x + leftRegion.width <= blue_mask.cols &&
                leftRegion.y + leftRegion.height <= blue_mask.rows) {

                Mat leftROI = blue_mask(leftRegion);
                if (countNonZero(leftROI) > 0) {
                    rectangle(result, rect, Scalar(0, 255, 0), 4);
                    cout << "Număr de înmatriculare posibil la: " << rect << endl;
                }
                }
        }
    }


    // Afisare rezultate intermediare
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
