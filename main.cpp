#include <iostream>
#include "src/functions.h"
#include "testing/testing.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

String path = "images/image3.jpeg";

int main() {

    // 1. Citim imaginea
    Mat source = imread(path);

    // 2. Convertim in grayscale si pastram o masca albastra pentru bara din stanga
    Mat gray = convertToGrayscale(source), blue_mask = extractBlueMask(source);

    // 3. Aplicam blur gaussian
    Mat blurred = gaussian_blur_filter(gray);

    // 4. Aplicam filtru median
    Mat medianed = median_filter(blurred, median3);

    // 5. Binarizare (threshold automatizat)
    edge_image_values values_img = compute_edge_values(medianed);
    int* histogram = compute_histogram_naive(medianed);
    int th = compute_bimodal_threshold(values_img, histogram, 0.1);
    Mat binary = apply_bimodal_thresholding(medianed, th);

    // 6. Detectie margini (Canny)
    Mat edges = cannyEdgeDetection(binary, 50, 150);

    // 7. Lista obiecte conexe
    vector<vector<Point>> objects = extract_all_objects(edges);

    // 8. Trasare manuala pentru ground truth
    draw_ground_truth(path, "testing/ground_truth.txt");

    // 8. Detectare placuțe dupa aspect ratio si dimensiune
    Mat result = source.clone();
    Mat plate;
    //pt testing
    vector<Rect> detections;
    //pana aici
    for (const auto& object : objects) {
        Rect rect = compute_bounding_box(object);
        double aspect = (double)rect.width / rect.height;

        if (aspect > 4.0 && aspect < 5.2 && compute_area(rect) > 3000) {
            Rect leftRegion(max(0, rect.x - rect.width), rect.y, rect.width, rect.height);

            if (leftRegion.x >= 0 && leftRegion.y >= 0 && leftRegion.x + leftRegion.width <= blue_mask.cols && leftRegion.y + leftRegion.height <= blue_mask.rows) {

                Mat leftROI = blue_mask(leftRegion);
                if (countNonZero(leftROI) > 0) {
                    //pt testing
                    detections.push_back(rect);
                    //pana aici
                    rectangle(result, rect, Scalar(0, 255, 0), 4);

                    plate = extract_license_plate(source, rect);
                }
            }
        }
    }

    imwrite("images/output/result.jpg", result);

    //pt testing
    vector<Rect> groundTruths = read_ground_truth("testing/ground_truth.txt");
    evaluate_detections(detections, groundTruths);
    //pana aici

    //Afisare rezultate intermediare
    namedWindow("Original", WINDOW_NORMAL);
    resizeWindow("Original", 600, 400);
    imshow("Original", source);

    namedWindow("Blue Mask", WINDOW_NORMAL);
    resizeWindow("Blue Mask", 600, 400);
    imshow("Blue Mask", blue_mask);

    namedWindow("Blurred", WINDOW_NORMAL);
    resizeWindow("Blurred", 600, 400);
    imshow("Blurred", blurred);

    namedWindow("Medianed", WINDOW_NORMAL);
    resizeWindow("Medianed", 600, 400);
    imshow("Medianed", medianed);

    namedWindow("Binary", WINDOW_NORMAL);
    resizeWindow("Binary", 600, 400);
    imshow("Binary", binary);

    namedWindow("Edges", WINDOW_NORMAL);
    resizeWindow("Edges", 600, 400);
    imshow("Edges", edges);

    namedWindow("Detected Plate", WINDOW_NORMAL);
    resizeWindow("Detected Plate", 600, 400);
    imshow("Detected Plate", plate);

    namedWindow("Framed Plate", WINDOW_NORMAL);
    resizeWindow("Framed Plate", 600, 400);
    imshow("Framed Plate", result);

    waitKey(0);
    return 0;
}
