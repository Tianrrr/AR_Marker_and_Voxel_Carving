#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <sstream>


using namespace std;

class voxel {
public:
    double X;
    double Y;
    double Z;
};

class PixelPos {
public:
    int x;
    int y;
};

class RGBA{
public:
    int R;
    int G;
    int B;
    int A;
};


class Boundingbox {
public:
    double startX;
    double startY;
    double startZ;
    double XWidth;
    double YHeight;
    double ZDepth;
};


class Carving
{
public:
    Carving (int dim, vector<cv::Mat> images, vector<cv::Mat> silhouettes, vector<cv::Mat> projectionMatrices)
    {
        
        
        VOXEL_DIM = dim;
        VOXEL_SIZE = VOXEL_DIM*VOXEL_DIM*VOXEL_DIM;
        VOXEL_SLICE = VOXEL_DIM*VOXEL_DIM;
        BACKGROUND = 0;

        vProjMatrix = projectionMatrices;
        vSrc = images;
        vSilhouette = silhouettes;

        

        double xmin = 0.0626, ymin = 0.1118, zmin = -0.0388;
        double xmax = 0.119, ymax = 0.1598, zmax = 0;
                
        // double Boxwidth = abs(xmax-xmin)*1.15;
        // double Boxheight = abs(ymax-ymin)*1.15;
        // double Boxdepth = abs(zmax-zmin)*1.05;

        Box.startX = xmin;
        Box.startY = ymin;
        Box.startZ = zmin;

        // cout << Box.startX << " " << Box.startY << " " << Box.startZ << endl;

        Box.XWidth = (xmax-xmin)/VOXEL_DIM;
        Box.YHeight = (ymax-ymin)/VOXEL_DIM;
        Box.ZDepth = (ymax-ymin)/VOXEL_DIM;

        // cout << "x: "  <<Box.XWidth << endl;
        // cout << "y: "  <<Box.YHeight << endl;
        // cout << "z: " <<Box.ZDepth << endl;

        pointNum = 0;

        vector<bool> vTemp(VOXEL_SIZE, 1);
        vMask = vTemp;
    }


    void RunCarving()
    {
        double timex = static_cast<double>(cv::getTickCount());
        cout << "Processing image....." << endl;
        for (int i=0; i<VOXEL_DIM; ++i) 
        {
            for (int j=0; j<VOXEL_DIM; ++j)
                {
                for (int k=0; k<VOXEL_DIM; ++k) 
                {
                    voxel v;
                    v.X = Box.startX + i * Box.XWidth;
                    v.Y = Box.startY + j * Box.YHeight;
                    v.Z = Box.startZ + k * Box.ZDepth;

                    int visibility = (int)vSilhouette.size();

                    for(int img=0; img<vSilhouette.size();++img)
                    {

                        PixelPos im = Project3Dto2D(vProjMatrix[img], v);

                        if (im.x <= 0 || im.y <= 0 || im.x >= vSilhouette[img].size().width|| im.y >= vSilhouette[img].size().height) 
                        {
                                visibility--;
                        }
                        else
                        {
                            if (vSilhouette[img].at<uchar>(im.y, im.x) == BACKGROUND)
                            {
                                visibility--;
                            }
                        }

                    }
                    if (visibility<(int)vSilhouette.size()-1)
                    {
                        vMask[i*VOXEL_SLICE+j*VOXEL_DIM+k] = 0;
                    }      
                }
            }
        }
        for (int i=0; i<VOXEL_DIM; i++) 
        {
            for (int j=0; j<VOXEL_DIM; j++)
             {
                for (int k=0; k<VOXEL_DIM; k++) 
                {  
                    if (vMask[i*VOXEL_SLICE+j*VOXEL_DIM+k])
                    {
                        double X = Box.startX + i * Box.XWidth;
                        double Y = Box.startY + j * Box.YHeight;
                        double Z = Box.startZ + k * Box.ZDepth;
                

                        voxel point = {X, Y, Z};

                        pointCloud.push_back(point);
                    }
                    
                }
            }
        }
        cout << "Carving finished" << endl;
        timex = ((double)cv::getTickCount() - timex) / cv::getTickFrequency();

        cout << "time for carving: " << timex << "s" << endl ;
    }
    
    //threshold for color consistency between 2 pixels onto 2 neigboring images which are projected by the same voxel
    void ColorRendering(int thres)
    {
        double timex = static_cast<double>(cv::getTickCount());

        cout << "Color rendering....." << endl;
        RGBA colorIntial = {BACKGROUND, BACKGROUND, BACKGROUND, BACKGROUND};
        vector<RGBA> v((int)pointCloud.size(), colorIntial);
        pointColor = v;

        for(int img=0; img<(int)vSilhouette.size();++img)
        {
            int next_img = img+1;
            if (next_img == (int)vSilhouette.size())
            {
                next_img = 0;
            }

            for (int i=0;i<(int)pointCloud.size();i++)
            {   
                PixelPos im = Project3Dto2D(vProjMatrix[img], pointCloud[i]);
                PixelPos im_next = Project3Dto2D(vProjMatrix[next_img], pointCloud[i]);

                if (ColorConsistency(im, im_next, img, next_img, thres) && pointColor[i].A == 0)
                {
                    int pixelB = vSrc[img].at<cv::Vec3b>(im.y,im.x)[0];
                    int pixelG = vSrc[img].at<cv::Vec3b>(im.y,im.x)[1];
                    int pixelR = vSrc[img].at<cv::Vec3b>(im.y,im.x)[2];
                    int pixelA = 255;

                    RGBA color = {pixelR, pixelG, pixelB, pixelA}; 
                    pointColor[i] = color;
                    pointNum++;
                }       
            }
       }
       cout << "Color rendering finished" << endl;
       timex = ((double)cv::getTickCount() - timex) / cv::getTickFrequency();
        cout << "time for rendering: " << timex << "s" << endl ;

    }

    vector<voxel>& get_Pos()
    {   
        for (int i=0; (int)i<pointCloud.size();++i)
        {
            if (pointColor[i].A != 0)
            {
                finalCloud.push_back(pointCloud[i]);
            }
        }
        return finalCloud;

    }

    vector<RGBA>& get_Color()
    {   
        for (int i=0; (int)i<pointCloud.size();++i)
        {
            if (pointColor[i].A != 0)
            {
                finalColor.push_back(pointColor[i]);
            }
        }
        return finalColor;
    }
    
    //x y z R G B A
    void SaveCloud()
    {   
        double timex = static_cast<double>(cv::getTickCount());

        cout <<"Saving Cloud in Cloud.off....." << endl;
        ofstream ofs;
        ofs.open("../Cloud.off", ios::out);
        ofs << "COFF" << endl;
        ofs << "# numVertices numFaces numEdges" << endl;
        ofs << pointNum << " " << 0 << " " << 0 << endl;
        ofs << "# list of vertices" << endl;
        ofs << "# X Y Z R G B A" << endl;

        for (int i=0;i<(int)pointCloud.size();i++)
        {   
            if (pointColor[i].A != 0)
            {
                ofs << pointCloud[i].X << " "
                    << pointCloud[i].Y << " "
                    << pointCloud[i].Z << " "
                    << pointColor[i].R << " "
                    << pointColor[i].G << " "
                    << pointColor[i].B << " "
                    << pointColor[i].A << endl;
            }       

        }
        ofs.close();
        timex = ((double)cv::getTickCount() - timex) / cv::getTickFrequency();
        cout << "time for saving: " << timex << "s" << endl ;
    }

    void SaveSilhouette()
    {
        int index = 0;
        for(vector<cv::Mat>::iterator iter = vSilhouette.begin(); iter!=vSilhouette.end(); iter++)
        {
            cv::imwrite("../Silhouette/silhouette_"+to_string(index)+".jpg", *iter );
            index++;
        }

    }


private:
    int Distance(PixelPos im1, PixelPos im2, int index1, int index2)
    {
        int DistanceB = vSrc[index1].at<cv::Vec3b>(im1.y,im1.x)[0] - vSrc[index2].at<cv::Vec3b>(im2.y,im2.x)[0];
        int DistanceG = vSrc[index1].at<cv::Vec3b>(im1.y,im1.x)[1] - vSrc[index2].at<cv::Vec3b>(im2.y,im2.x)[1];
        int DistanceR = vSrc[index1].at<cv::Vec3b>(im1.y,im1.x)[2] - vSrc[index2].at<cv::Vec3b>(im2.y,im2.x)[2];

        int dis= DistanceB*DistanceB + DistanceG*DistanceG + DistanceR*DistanceR;
        return dis;
    }

    bool ColorConsistency(PixelPos im1, PixelPos im2, int index1, int index2, int thres)
    {
        if (Distance(im1, im2, index1, index2) <= thres)
        {
            return true;
        }
        return false;
    }


    PixelPos Project3Dto2D(cv::Mat projMatrix, voxel v) 
    {
        
        PixelPos im;

        
        double z =   projMatrix.at<double>(2, 0) * v.X +
                    projMatrix.at<double>(2, 1) * v.Y +
                    projMatrix.at<double>(2, 2) * v.Z +
                    projMatrix.at<double>(2, 3);

        im.y =    (projMatrix.at<double>(1, 0) * v.X +
                projMatrix.at<double>(1, 1) * v.Y +
                projMatrix.at<double>(1, 2) * v.Z +
                projMatrix.at<double>(1, 3)) / z;
        
        im.x =    (projMatrix.at<double>(0, 0) * v.X +
                projMatrix.at<double>(0, 1) * v.Y +
                projMatrix.at<double>(0, 2) * v.Z +
                projMatrix.at<double>(0, 3)) / z;

        
        
        
        return im;
    }


    vector<cv::Mat> vProjMatrix;
    vector<cv::Mat> vSrc;
    vector<cv::Mat> vSilhouette;
    // vector<cv::Mat> vGray;
    vector<voxel> pointCloud;
    vector<RGBA> pointColor; 
    vector<voxel> finalCloud;
    vector<RGBA> finalColor; 
    vector<bool> vMask;
    Boundingbox Box;
    int pointNum;
    int VOXEL_DIM;
    int VOXEL_SIZE;
    int VOXEL_SLICE;
    int BACKGROUND;
};

