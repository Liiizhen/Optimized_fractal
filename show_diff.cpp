#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <folder_name>" << std::endl;
        return 1;
    }
    std::string s = argv[1];
    std::string csv_path = "output_" + s + ".csv";
    std::ifstream ifs(csv_path);
    if (!ifs.is_open()) {
        std::cerr << "Failed to open csv file: " << csv_path << std::endl;
        return 1;
    }
    std::string line;
    std::getline(ifs, line); // skip header
    while (std::getline(ifs, line)) {
        std::stringstream ss(line);
        std::string filename, opencv_count_str, opencv_time, nano_count_str, nano_time;
        std::getline(ss, filename, ',');
        std::getline(ss, opencv_count_str, ',');
        std::getline(ss, opencv_time, ',');
        std::getline(ss, nano_count_str, ',');
        std::getline(ss, nano_time, ',');
        if (opencv_count_str != nano_count_str) {
            std::string opencv_img_path = "fractal_set/" + s + "/opencv/" + filename;
            std::string nano_img_path = "fractal_set/" + s + "/nano/" + filename;
            cv::Mat opencv_img = cv::imread(opencv_img_path);
            cv::Mat nano_img = cv::imread(nano_img_path);
            if (opencv_img.empty() || nano_img.empty()) {
                std::cerr << "Failed to read image: " << opencv_img_path << " or " << nano_img_path << std::endl;
                continue;
            }
            int target_width = 800;
            double scale1 = (double)target_width / opencv_img.cols;
            double scale2 = (double)target_width / nano_img.cols;
            int h1 = (int)(opencv_img.rows * scale1);
            int h2 = (int)(nano_img.rows * scale2);
            cv::resize(opencv_img, opencv_img, cv::Size(target_width, h1));
            cv::resize(nano_img, nano_img, cv::Size(target_width, h2));
            
            int final_height = std::min(opencv_img.rows, nano_img.rows);
            if (opencv_img.rows != final_height)
                cv::resize(opencv_img, opencv_img, cv::Size(target_width, final_height));
            if (nano_img.rows != final_height)
                cv::resize(nano_img, nano_img, cv::Size(target_width, final_height));
            cv::Mat concat_img;
            cv::hconcat(opencv_img, nano_img, concat_img);
            std::cout << "Showing: " << filename << " opencv_count=" << opencv_count_str << " nano_count=" << nano_count_str << std::endl;
            cv::imshow("opencv | nano", concat_img);
            cv::waitKey(0);
        }
    }
    return 0;
}
