#include <filesystem>
#include <iostream>
#include <chrono>
#include "nanofractal.h"
#include "opencv_fractal.h"
#include <fstream>

int main() {
    try {
        std::string imagePath = "data/test.png";
        cv::Mat imagemodel = cv::imread(imagePath);

        std::string distortion_672_504 = "data/distortion_672_504.jpg";
        std::string distortion_1008_756 = "data/distortion_1008_756.jpg";
        std::string distortion_1344_1008 = "data/distortion_1344_1008.jpg";
        std::string distortion_2016_1512 = "data/distortion_2016_1512.jpg";
        std::string distortion_4032_3024 = "data/distortion_4032_3024.jpg";
        std::string occ_672_504 = "data/occ_672_504.jpg";
        std::string occ_1008_756 = "data/occ_1008_756.jpg";
        std::string occ_1344_1008 = "data/occ_1344_1008.jpg";
        std::string occ_2016_1512 = "data/occ_2016_1512.jpg";
        std::string occ_4032_3024 = "data/occ_4032_3024.jpg";


        std::string inputImagePath = occ_4032_3024;

        cv::Mat image = cv::imread(inputImagePath);
        if (image.empty()) {
            return -1;
        }


        // cv::Mat bwimage;
        // if(image.channels()==3)
        //     cv::cvtColor(image,bwimage,cv::COLOR_BGR2GRAY);
        // else bwimage=image;

        std::filesystem::path inputPath(inputImagePath);

        // 1. nanofractal detection
        cv::Mat nanoImage = image.clone();
        nanofractal::FractalMarkerDetector nanoDetector;
        nanoDetector.setParams("FRACTAL_4L_6");
        std::vector<cv::Point3f> nanoPoints3D;
        std::vector<cv::Point2f> nanoPoints2D;
        auto nanoStart = std::chrono::high_resolution_clock::now();
        std::vector<nanofractal::FractalMarker> nanoMarkers = nanoDetector.detect(nanoImage, nanoPoints3D, nanoPoints2D);
        auto nanoEnd = std::chrono::high_resolution_clock::now();
        double nanoTime = std::chrono::duration<double, std::milli>(nanoEnd - nanoStart).count();

        std::cout << "Nano matched points number: " << nanoPoints3D.size() << ". " << std::endl;
        std::cout << "Nano detection time: " << nanoTime << " ms" << std::endl;
        for (const auto& marker : nanoMarkers) {
            marker.draw(nanoImage);
        }
        for (const auto& point : nanoPoints2D) {
            cv::circle(nanoImage, point, 5, cv::Scalar(0, 255, 0), cv::FILLED);
        }

        std::string nanoOutput = inputPath.parent_path().string() + "/nano_" + inputPath.filename().string();
        cv::imwrite(nanoOutput, nanoImage);
        std::cout << "Nano result saved to: " << nanoOutput << std::endl;

        // 2. opencvfractal detection
        cv::Mat opencvImage = image.clone();
        opencvfractal::FractalMarkerDetector opencvDetector;
        opencvDetector.setParams("FRACTAL_4L_6");
        std::vector<cv::Point3f> opencvPoints3D;
        std::vector<cv::Point2f> opencvPoints2D;

        auto opencvStart = std::chrono::high_resolution_clock::now();
        std::vector<opencvfractal::FractalMarker> opencvMarkers = opencvDetector.detect(opencvImage, opencvPoints3D, opencvPoints2D);
        auto opencvEnd = std::chrono::high_resolution_clock::now();
        double opencvTime = std::chrono::duration<double, std::milli>(opencvEnd - opencvStart).count();

        std::cout << "OpenCV matched points number: " << opencvPoints3D.size() << ". " << std::endl;
        std::cout << "OpenCV detection time: " << opencvTime << " ms" << std::endl;
        for (const auto& marker : opencvMarkers) {
            marker.draw(opencvImage);
        }
        for (const auto& point : opencvPoints2D) {
            cv::circle(opencvImage, point, 5, cv::Scalar(0, 255, 0), cv::FILLED);
        }

        cv::Mat imgWithP2D = image.clone();
        cv::Mat imageWithP3D = imagemodel.clone();

        if (imgWithP2D.channels() == 1) {
            cv::cvtColor(imgWithP2D, imgWithP2D, cv::COLOR_GRAY2BGR);
        }
        if (imageWithP3D.channels() == 1) {
            cv::cvtColor(imageWithP3D, imageWithP3D, cv::COLOR_GRAY2BGR);
        }
// ...existing code...
std::vector<cv::Point> mappedP3DPoints;

<<<<<<< HEAD
for (size_t i = 0; i < opencvPoints2D.size(); ++i) {
    cv::circle(imgWithP2D, opencvPoints2D[i], 3, cv::Scalar(0, 255, 0), cv::FILLED);

    int imgWidth = imagemodel.cols;
    int imgHeight = imagemodel.rows;
    int x = static_cast<int>((opencvPoints3D[i].x + 1.0) * 0.5 * imgWidth);
    int y = static_cast<int>((1.0 - (opencvPoints3D[i].y + 1.0) * 0.5) * imgHeight);
    cv::Point mappedPt(x, y);
    mappedP3DPoints.push_back(mappedPt);

    if (x >= 0 && x < imgWidth && y >= 0 && y < imgHeight) {
        cv::circle(imageWithP3D, mappedPt, 5, cv::Scalar(0, 0, 255), cv::FILLED);
    }
=======
        std::vector<cv::Point> mappedP3DPoints;

        for (size_t i = 0; i < opencvPoints2D.size(); ++i) {
            cv::circle(imgWithP2D, opencvPoints2D[i], 3, cv::Scalar(0, 255, 0), cv::FILLED);
        
            int imgWidth = imagemodel.cols;
            int imgHeight = imagemodel.rows;
            int x = static_cast<int>((opencvPoints3D[i].x + 1.0) * 0.5 * imgWidth);
            int y = static_cast<int>((1.0 - (opencvPoints3D[i].y + 1.0) * 0.5) * imgHeight);
            cv::Point mappedPt(x, y);
            mappedP3DPoints.push_back(mappedPt);
        
            if (x >= 0 && x < imgWidth && y >= 0 && y < imgHeight) {
                cv::circle(imageWithP3D, mappedPt, 5, cv::Scalar(0, 0, 255), cv::FILLED);
            }
        }
>>>>>>> 5c5145ee5a3ad07751bc95e018c85f35c85da557

    // 每10个点展示一次窗口
    if ((i + 1) % 10 == 0 || i == opencvPoints2D.size() - 1) {
        cv::Mat tempImgWithP2D = imgWithP2D.clone();
        cv::Mat tempImageWithP3D = imageWithP3D.clone();
        cv::Mat tempCombinedImage;
        if (tempImgWithP2D.type() != tempImageWithP3D.type()) {
            tempImageWithP3D.convertTo(tempImageWithP3D, tempImgWithP2D.type());
        }
        double scaleP2D = 1.0, scaleP3D = 1.0;
        if (tempImgWithP2D.rows != tempImageWithP3D.rows) {
            int targetHeight = std::min(tempImgWithP2D.rows, tempImageWithP3D.rows);
            scaleP2D = static_cast<double>(targetHeight) / tempImgWithP2D.rows;
            scaleP3D = static_cast<double>(targetHeight) / tempImageWithP3D.rows;
            int newWidthP2D = static_cast<int>(tempImgWithP2D.cols * scaleP2D);
            int newWidthP3D = static_cast<int>(tempImageWithP3D.cols * scaleP3D);
            cv::resize(tempImgWithP2D, tempImgWithP2D, cv::Size(newWidthP2D, targetHeight));
            cv::resize(tempImageWithP3D, tempImageWithP3D, cv::Size(newWidthP3D, targetHeight));
        }
        cv::hconcat(tempImgWithP2D, tempImageWithP3D, tempCombinedImage);

        int offsetX = tempImgWithP2D.cols;
        for (size_t j = 0; j <= i; ++j) {
            cv::Point pt1 = cv::Point(
                static_cast<int>(opencvPoints2D[j].x * scaleP2D),
                static_cast<int>(opencvPoints2D[j].y * scaleP2D)
            );
<<<<<<< HEAD
            cv::Point pt2 = mappedP3DPoints[j] + cv::Point(offsetX, 0);
            cv::line(tempCombinedImage, pt1, pt2, cv::Scalar(0, 0, 255), 1);
=======
            cv::Point pt2 = mappedP3DPoints[i] + cv::Point(offsetX, 0);
            cv::line(combinedImage, pt1, pt2, cv::Scalar(0, 0, 255), 1);
>>>>>>> 5c5145ee5a3ad07751bc95e018c85f35c85da557
        }

        cv::namedWindow("P2D and P3D Points", cv::WINDOW_NORMAL);
        cv::resizeWindow("P2D and P3D Points", 2000, 1200);
        cv::imshow("P2D and P3D Points", tempCombinedImage);
        cv::waitKey(0);
        cv::destroyWindow("P2D and P3D Points");
    }
}
// ...existing code...
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
