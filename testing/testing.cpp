#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "testing.h"
using namespace std;
using namespace cv;

static vector<Rect> boxes;
static bool drawing = false;
static Point startPoint;
static Mat img, temp;

void mouseHandler(int event, int x, int y, int flags, void* userdata) {
    if (event == EVENT_LBUTTONDOWN) {
        startPoint = Point(x, y);
        drawing = true;
    } else if (event == EVENT_MOUSEMOVE && drawing) {
        temp = img.clone();
        Rect box(startPoint, Point(x, y));
        rectangle(temp, box, Scalar(0, 255, 0), 2);
        imshow("Select ground truth", temp);
    } else if (event == EVENT_LBUTTONUP && drawing) {
        drawing = false;
        Rect box(startPoint, Point(x, y));
        boxes.push_back(box);
        rectangle(img, box, Scalar(0, 255, 0), 2);
        imshow("Select ground truth", img);
        cout << "Adaugat: " << box << endl;
    }
}

void draw_ground_truth(const string& imagePath, const string& outputFilePath) {
    boxes.clear();
    img = imread(imagePath);
    temp = img.clone();
    namedWindow("Select ground truth", WINDOW_NORMAL);
    resizeWindow("Select ground truth", 800, 600);
    setMouseCallback("Select ground truth", mouseHandler, NULL);
    imshow("Select ground truth", img);

    cout << "Deseneaza dreptunghiuri cu mouse-ul. Apasa ESC pentru a salva si iesi.\n";
    while (true) {
        int key = waitKey(0);
        if (key == 27 || getWindowProperty("NumeFereastra", cv::WND_PROP_VISIBLE) < 1) {
            break; // ESC sau inchidem fereastra
        }
    }

    ofstream out(outputFilePath);

    for (const auto& box : boxes) {
        out << box.x << " " << box.y << " " << box.width << " " << box.height << endl;
    }

    out.close();
    cout << "Am salvat " << boxes.size() << " ground truth box-uri in: " << outputFilePath << endl;
}

vector<Rect> read_ground_truth(const string& filePath) {
    vector<Rect> groundTruths;
    ifstream in(filePath);

    int x, y, w, h;
    while (in >> x >> y >> w >> h) {
        groundTruths.emplace_back(x, y, w, h);
    }

    in.close();
    return groundTruths;
}

double compute_IoU(const Rect& a, const Rect& b) {
    int x1 = max(a.x, b.x);
    int y1 = max(a.y, b.y);
    int x2 = min(a.x + a.width, b.x + b.width);
    int y2 = min(a.y + a.height, b.y + b.height);

    int interArea = max(0, x2 - x1) * max(0, y2 - y1);
    int unionArea = a.area() + b.area() - interArea;

    if (unionArea == 0) return 0.0;
    return (double)interArea / unionArea;
}

void evaluate_detections(const vector<Rect>& detections, const vector<Rect>& groundTruths) {
    const double IOU_THRESHOLD = 0.65;
    int TP = 0, FP = 0, FN = 0;

    vector<bool> gt_matched(groundTruths.size(), false);

    for (const auto& det : detections) {
        bool matched = false;
        for (size_t i = 0; i < groundTruths.size(); ++i) {
            double IOU_VALUE = compute_IoU(det, groundTruths[i]);
            if (!gt_matched[i] && IOU_VALUE >= IOU_THRESHOLD) {
                TP++;
                gt_matched[i] = true;
                matched = true;
                cout<< "IOU Value: "<< IOU_VALUE << endl;
                break;
            }
        }
        if (!matched) FP++;
    }

    for (bool matched : gt_matched) {
        if (!matched) FN++;
    }

    double precision = TP + FP == 0 ? 0 : (double)TP / (TP + FP);
    double recall    = TP + FN == 0 ? 0 : (double)TP / (TP + FN);
    double accuracy  = groundTruths.empty() ? 0 : (double)TP / groundTruths.size();

    cout << "--- Evaluare detectie ---" << endl;
    cout << "True Positives: " << TP << endl;
    cout << "False Positives: " << FP << endl;
    cout << "False Negatives: " << FN << endl;
    cout << "Precision: " << precision << endl;
    cout << "Recall:    " << recall << endl;
    cout << "Accuracy:  " << accuracy << endl;
}