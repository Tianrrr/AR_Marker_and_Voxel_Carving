#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include "multiImage.h"
// #include "voxel.h"
#include "Eigen.h"
#include "carving2.cpp"

// read list.txt to get file name of input images
void readTXT(std::string inputDataDir, std::vector<std::string> &fileNames) {
    std::ifstream input(inputDataDir + "list.txt");
    if (!input.is_open()) {
        std::cout << "Error: error when reading 'list.txt'" << std::endl;
        throw -1;
    } else {
        std::string line;
        while (!input.eof()) {
            std::getline(input, line);
            fileNames.push_back(inputDataDir + line);
        }
    }
}

// save multiple images in specific directory
void saveImages(std::string dstDir, std::vector<cv::Mat> images) {
    std::string fileName;
    for (int i = 0; i < images.size(); i++) {
        if (i < 9) {
            fileName = "0" + std::to_string(i+1) + ".png";
        } else {
            fileName = std::to_string(i+1) + ".png";
        }
        cv::imwrite(dstDir + fileName, images[i]);
    }
}

int main()
{
    // define file path
    std::string inputDataDir = "../InputData/tuzi/";
    std::string outputDataDir = "../OutputData/tuzi/";
    // read image path
    std::vector<std::string> fileNames;
    readTXT(inputDataDir, fileNames);
    // create MultiImage class, get all information
    MultiImage images(fileNames, 1);
//    saveImages(outputDataDir+"silhouette/iter3/",images.silhouettes);
//    images.extractForeground();
//    saveImages(outputDataDir+"foreground/iter3/", images.foregrounds);

    // define bounding box
    Eigen::Vector3d startPoint(0.0626,0.1118,-0.0388);
    Eigen::Vector3d endPoint(0.119,0.1598,0);
    int n = 200;
    // Voxel space(n,n,n, startPoint, endPoint, images.images, images.silhouettes, images.p_Matrices);
    // space.carve(1);
    // space.colorRender(4);
    // space.writeCenterPoints(outputDataDir + "Iter_0_c.off", true);
//     space.writeCenterPoints(outputDataDir + "iter_0_c_voxel_thres1.off", true);
//    std::cout<<space.n_centerPoints<<" "<<space.n_vertices<<" "<<space.n_faces<<std::endl;  


    Carving c(n, images.images, images.silhouettes, images.p_Matrices);
    c.RunCarving();
    c.ColorRendering(3);
    c.SaveCloud();
    c.SaveSilhouette();






    return 0;
}