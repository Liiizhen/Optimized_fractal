#include <filesystem>
#include <iostream>
#include <chrono>
#include "nanofractal.h"
#include "opencv_fractal.h"
#include <fstream>

int main() {
    try {
        std::string imagePath = "/mnt/d/code/nano/Fractal/data/input/test.png";
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

        std::vector<cv::Point> mappedP3DPoints;

        for (size_t i = 0; i < opencvPoints2D.size(); ++i) {
            cv::circle(imgWithP2D, opencvPoints2D[i], 3, cv::Scalar(0, 255, 0), cv::FILLED); // 绿色点
        
            int imgWidth = imagemodel.cols;
            int imgHeight = imagemodel.rows;
            int x = static_cast<int>((opencvPoints3D[i].x + 1.0) * 0.5 * imgWidth);  // 将 -1 到 1 映射到 0 到 imgWidth
            int y = static_cast<int>((1.0 - (opencvPoints3D[i].y + 1.0) * 0.5) * imgHeight);  // 将 -1 到 1 映射到 0 到 imgHeight
            cv::Point mappedPt(x, y);
            mappedP3DPoints.push_back(mappedPt);
        
            if (x >= 0 && x < imgWidth && y >= 0 && y < imgHeight) {
                cv::circle(imageWithP3D, mappedPt, 5, cv::Scalar(0, 0, 255), cv::FILLED); // 红色点
            }
        }

        cv::Mat combinedImage;
        if (imgWithP2D.type() != imageWithP3D.type()) {
            imageWithP3D.convertTo(imageWithP3D, imgWithP2D.type());
        }
        double scaleP2D = 1.0, scaleP3D = 1.0;
        if (imgWithP2D.rows != imageWithP3D.rows) {
            int targetHeight = std::min(imgWithP2D.rows, imageWithP3D.rows);
            scaleP2D = static_cast<double>(targetHeight) / imgWithP2D.rows;
            scaleP3D = static_cast<double>(targetHeight) / imageWithP3D.rows;
            int newWidthP2D = static_cast<int>(imgWithP2D.cols * scaleP2D);
            int newWidthP3D = static_cast<int>(imageWithP3D.cols * scaleP3D);
            cv::resize(imgWithP2D, imgWithP2D, cv::Size(newWidthP2D, targetHeight));
            cv::resize(imageWithP3D, imageWithP3D, cv::Size(newWidthP3D, targetHeight));
        }
        cv::hconcat(imgWithP2D, imageWithP3D, combinedImage);

        int offsetX = imgWithP2D.cols;
        for (size_t i = 0; i < opencvPoints2D.size(); ++i) {
            cv::Point pt1 = cv::Point(
                static_cast<int>(opencvPoints2D[i].x * scaleP2D),
                static_cast<int>(opencvPoints2D[i].y * scaleP2D)
            );
            cv::Point pt2 = mappedP3DPoints[i] + cv::Point(offsetX, 0);
            cv::line(combinedImage, pt1, pt2, cv::Scalar(0, 0, 255), 1); // 红色线
        }

        cv::namedWindow("P2D and P3D Points", cv::WINDOW_NORMAL);
        cv::resizeWindow("P2D and P3D Points", 2000, 1200);

        cv::imshow("P2D and P3D Points", combinedImage);

        cv::waitKey(0);
        cv::destroyWindow("P2D and P3D Points");
        std::string opencvOutput = inputPath.parent_path().string() + "/opencv_" + inputPath.filename().string();
        cv::imwrite(opencvOutput, opencvImage);
        std::cout << "OpenCV result saved to: " << opencvOutput << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}