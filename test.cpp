#include <filesystem>
#include <iostream>
#include <chrono>
#include "nanofractal.h"
#include "opencv_fractal.h"
#include <fstream>

int main() {
    try {

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
        std::string opencvOutput = inputPath.parent_path().string() + "/opencv_" + inputPath.filename().string();
        cv::imwrite(opencvOutput, opencvImage);
        std::cout << "OpenCV result saved to: " << opencvOutput << std::endl;


    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}