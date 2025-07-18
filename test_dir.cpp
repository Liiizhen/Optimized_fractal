#include <filesystem>
#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "nanofractal.h"
#include "opencv_fractal.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <directory_path>" << std::endl;
        return 1;
    }
    std::string dirPath = argv[1];
    std::filesystem::path folder(dirPath);
    if (!std::filesystem::exists(folder) || !std::filesystem::is_directory(folder)) {
        std::cerr << "Invalid directory: " << dirPath << std::endl;
        return 1;
    }

    // Output file
    std::string outputFile = "output_" + folder.filename().string() + ".csv";
    std::ofstream ofs(outputFile);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open output file: " << outputFile << std::endl;
        return 1;
    }
    ofs << "filename,opencv_count,opencv_time_ms,nano_count,nano_time_ms" << std::endl;

    for (const auto& entry : std::filesystem::directory_iterator(folder)) {
        if (!entry.is_regular_file()) continue;
        std::string filePath = entry.path().string();
        if (entry.path().extension() != ".jpg") continue;

        cv::Mat image = cv::imread(filePath);
        if (image.empty()) {
            std::cerr << "Failed to read image: " << filePath << std::endl;
            continue;
        }

        // OpenCV version
        cv::Mat opencvImage = image.clone();
        opencvfractal::FractalMarkerDetector opencvDetector;
        opencvDetector.setParams("FRACTAL_4L_6");
        std::vector<cv::Point3f> opencvPoints3D;
        std::vector<cv::Point2f> opencvPoints2D;
        auto opencvStart = std::chrono::high_resolution_clock::now();
        std::vector<opencvfractal::FractalMarker> opencvMarkers = opencvDetector.detect(opencvImage, opencvPoints3D, opencvPoints2D);
        auto opencvEnd = std::chrono::high_resolution_clock::now();
        double opencvTime = std::chrono::duration<double, std::milli>(opencvEnd - opencvStart).count();

        for (const auto& marker : opencvMarkers) {
            marker.draw(opencvImage);
        }
        for (const auto& point : opencvPoints2D) {
            cv::circle(opencvImage, point, 5, cv::Scalar(0, 255, 0), cv::FILLED);
        }
        // 保存到 opencv 文件夹
        std::filesystem::path opencvDir = entry.path().parent_path() / "opencv";
        if (!std::filesystem::exists(opencvDir)) {
            std::filesystem::create_directory(opencvDir);
        }
        std::string opencvOutput = (opencvDir / entry.path().filename()).string();
        cv::imwrite(opencvOutput, opencvImage);

        // Nano version
        cv::Mat nanoImage = image.clone();
        nanofractal::FractalMarkerDetector nanoDetector;
        nanoDetector.setParams("FRACTAL_4L_6");
        std::vector<cv::Point3f> nanoPoints3D;
        std::vector<cv::Point2f> nanoPoints2D;
        auto nanoStart = std::chrono::high_resolution_clock::now();
        std::vector<nanofractal::FractalMarker> nanoMarkers = nanoDetector.detect(nanoImage, nanoPoints3D, nanoPoints2D);
        auto nanoEnd = std::chrono::high_resolution_clock::now();
        double nanoTime = std::chrono::duration<double, std::milli>(nanoEnd - nanoStart).count();

        for (const auto& marker : nanoMarkers) {
            marker.draw(nanoImage);
        }
        for (const auto& point : nanoPoints2D) {
            cv::circle(nanoImage, point, 5, cv::Scalar(0, 255, 0), cv::FILLED);
        }
        // 保存到 nano 文件夹
        std::filesystem::path nanoDir = entry.path().parent_path() / "nano";
        if (!std::filesystem::exists(nanoDir)) {
            std::filesystem::create_directory(nanoDir);
        }
        std::string nanoOutput = (nanoDir / entry.path().filename()).string();
        cv::imwrite(nanoOutput, nanoImage);

        ofs << entry.path().filename().string() << ","
            << opencvPoints3D.size() << "," << opencvTime << ","
            << nanoPoints3D.size() << "," << nanoTime << std::endl;
    }

    ofs.close();
    // std::cout << "Results saved to: " << outputFile << std::endl;
    return 0;
}
