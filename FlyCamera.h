#include <iostream>
#include "FlyCapture2.h"      
#include <opencv2/opencv.hpp>
// using namespace cv;
// using namespace std;
using namespace FlyCapture2; 

class FlyCamera
{
public:
    bool checkError(FlyCapture2::Error err);
    bool camera_init();
    bool camera_capture(int cam_num, cv::Mat& src);
    bool camera_capture2(int cam_num, cv::Mat& src2);
    bool camera_kill(int cam_num);

    Camera* pCameras;
    unsigned int camera_num;
    unsigned int cam_serial_num[4];

private:
    FlyCapture2::Error err;
    BusManager mgr;
    PGRGuid guid;
    CameraInfo camInfo;
    TriggerMode trigMode;
    EmbeddedImageInfo imageInfo;
    Image rawImage, rawImage2;
    Image rgbImage, rgbImage2;
    unsigned int rowBytes, rowBytes2;
};