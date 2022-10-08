#include "FlyCamera.h"

bool FlyCamera::checkError(FlyCapture2::Error err)
{
    if (err != PGRERROR_OK)
    {
        err.PrintErrorTrace();
        std::cout << std::endl << "Press Enter to exit." << std::endl;
        delete[] pCameras;
        std::cin.ignore();
        return false;
    }
    return true;
}

bool FlyCamera::camera_init()
{
    err = mgr.GetNumOfCameras(&camera_num);
    if(err != PGRERROR_OK)
    {
        err.PrintErrorTrace();
        return false;
    }
    std::cout << "Number of cameras detected: " << camera_num << std::endl;

    pCameras = new FlyCapture2::Camera[camera_num];
    for(int i = 0; i < camera_num; i++)
    {
        err = mgr.GetCameraFromIndex(i, &guid);
        if (!checkError(err)) return false;

        // Connect to a camera
        err = pCameras[i].Connect(&guid);
        if (!checkError(err)) return false;

        // Get the camera information
        err = pCameras[i].GetCameraInfo(&camInfo);
        if (!checkError(err)) return false;

        // Get serial number of camera 
        err = mgr.GetCameraSerialNumberFromIndex(i, &cam_serial_num[i]);
        if (!checkError(err)) return false;

        // Turn trigger mode off
        trigMode.onOff = false;
        err = pCameras[i].SetTriggerMode(&trigMode);
        if (!checkError(err)) return false;

        // Turn Timestamp on
        imageInfo.timestamp.onOff = true;
        imageInfo.whiteBalance.available=true;
        imageInfo.whiteBalance.onOff=true;                                              
        err = pCameras[i].SetEmbeddedImageInfo(&imageInfo);
        if (!checkError(err)) return false;

        // Start streaming on camera
        err = pCameras[i].StartCapture();
        if (!checkError(err)) return false;

    }
    return true;
}

bool FlyCamera::camera_capture(int cam_num, cv::Mat& src)
{
    // 获取相机中原始图像
    err = pCameras[cam_num].RetrieveBuffer(&rawImage);
    if (err != PGRERROR_OK)
    {
        err.PrintErrorTrace();
        std::cin.ignore();
        return false;
    }
    // 转换为RGB图像
    rawImage.Convert( FlyCapture2::PIXEL_FORMAT_BGR, &rgbImage );
    // 转换为OpenCV格式的图像
    rowBytes = (double)rgbImage.GetReceivedDataSize()/(double)rgbImage.GetRows();
    src = cv::Mat(rgbImage.GetRows(), rgbImage.GetCols(), CV_8UC3, rgbImage.GetData(),rowBytes);
    return true;
}

bool FlyCamera::camera_capture2(int cam_num, cv::Mat& src2)
{
    // 获取相机中原始图像
    err = pCameras[cam_num].RetrieveBuffer(&rawImage2);
    if (err != PGRERROR_OK)
    {
        err.PrintErrorTrace();
        std::cin.ignore();
        return false;
    }
    // 转换为RGB图像
    rawImage2.Convert( FlyCapture2::PIXEL_FORMAT_BGR, &rgbImage2 );
    // 转换为OpenCV格式的图像
    rowBytes2 = (double)rgbImage2.GetReceivedDataSize()/(double)rgbImage2.GetRows();
    src2 = cv::Mat(rgbImage2.GetRows(), rgbImage2.GetCols(), CV_8UC3, rgbImage2.GetData(),rowBytes);
    return true;
}

bool FlyCamera::camera_kill(int cam_num)
{
    //停止捕获图像
    err = pCameras[cam_num].StopCapture();
    if (err != PGRERROR_OK)
    {
        err.PrintErrorTrace();
        std::cin.ignore();
        return false;
    }
    //断开相机连接
    pCameras[cam_num].Disconnect();
}