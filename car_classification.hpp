#include <iostream>
#include <sys/select.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <opencv2/opencv.hpp>

#define SENTRY_ADDR "192.168.1.128"

class Position
{
public:
    Position()
    {
        state = false;
        poi.x = 1000;
        poi.y = 1000;
    }
    void clearPoi()
    {
        last_state = state;
        last_poi.x = poi.x;
        last_poi.y = poi.y;
        state = false;
        poi.x = 1000; 
        poi.y = 1000;
    }

    bool state, last_state;
    cv::Point2f poi, last_poi;
};

class Number
{
public:
    void clearNumber()
    {
        one.clearPoi();
        two.clearPoi();
    }

    Position one;
    Position two;
};

class CarState
{
public:
    void clearCarState()
    {
        blue.clearNumber();
        red.clearNumber();
    }

    bool CarTracing(Position last, cv::Point2f poi_);
    void CarClassify(int car_num, double x, double y);

    Number blue;
    Number red;
};

bool CarState::CarTracing(Position last, cv::Point2f poi_)
{
    if(!last.last_state) return false;
    return true;
}

void CarState::CarClassify(int car_num, double x, double y)
{
    double px = (808 - x) / 100;
    double py = y / 100;
    switch(car_num)
    {
        case(2): 
        {
            //printf("蓝色一号车  ");
            blue.one.state = true;
            blue.one.poi.x = px;
            blue.one.poi.y = py;
            break;
        }
        case(3): 
        {
            //printf("蓝色二号车  ");
            blue.two.state = true;
            blue.two.poi.x = px;
            blue.two.poi.y = py;
            break;
        }
        case(0): 
        {
            //printf("红色一号车  ");
            red.one.state = true;
            red.one.poi.x = px;
            red.one.poi.y = py;
            break;
        }
        case(1): 
        {
            //printf("红色二号车  ");
            red.two.state = true;
            red.two.poi.x = px;
            red.two.poi.y = py;
            break;
        }
    }
    //printf("X: %f, Y: %f\n", px, py);
    return;
}

class _Socket_
{
public:
    struct CarMsg
    {
        bool blue_1_state;
        bool blue_2_state;
        bool red_1_state;
        bool red_2_state;
        double blue_1_x;
        double blue_1_y;
        double blue_2_x;
        double blue_2_y;
        double red_1_x;
        double red_1_y;
        double red_2_x;
        double red_2_y;
    }car_msg;
    void IntegrateInfo(CarState car_state);
    void clearMsg();
    void setSocket(const char* sentry_addr, int port_in);
    void sendSocket();
    void closeSocket();

    int sockfd, portIn;
    struct sockaddr_in addr;
    socklen_t  addr_len=sizeof(addr);
};

void _Socket_::IntegrateInfo(CarState car_state)
{
        car_msg.blue_1_state = car_state.blue.one.state;
        car_msg.blue_2_state = car_state.blue.two.state;
        car_msg.red_1_state = car_state.red.one.state;
        car_msg.red_2_state = car_state.red.two.state;
        car_msg.blue_1_x = (double)car_state.blue.one.poi.x;
        car_msg.blue_1_y = (double)car_state.blue.one.poi.y;
        car_msg.blue_2_x = (double)car_state.blue.two.poi.x;
        car_msg.blue_2_y = (double)car_state.blue.two.poi.y;
        car_msg.red_1_x = (double)car_state.red.one.poi.x;
        car_msg.red_1_y = (double)car_state.red.one.poi.y;
        car_msg.red_2_x = (double)car_state.red.two.poi.x;
        car_msg.red_2_y = (double)car_state.red.two.poi.y;
}

void _Socket_::setSocket(const char* sentry_addr, int port_in)
{
    portIn = port_in;
    // 创建socket
    sockfd = socket(PF_INET, SOCK_DGRAM, 0);
    if(-1==sockfd){
        return;
        puts("Failed to create socket");
    }

    // 设置地址与端口
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;       // Use IPV4
    addr.sin_port   = htons(portIn);    //
    addr.sin_addr.s_addr = inet_addr(sentry_addr);
}

void _Socket_::sendSocket()
{
    char buffer[128];
    memset(buffer, 0 ,128);
    sprintf(buffer, "%d %d %d %d %lf %lf %lf %lf %lf %lf %lf %lf", car_msg.blue_1_state, car_msg.blue_2_state, car_msg.red_1_state, 
        car_msg.red_2_state, car_msg.blue_1_x, car_msg.blue_1_y, car_msg.blue_2_x, car_msg.blue_2_y, car_msg.red_1_x, car_msg.red_1_y, 
        car_msg.red_2_x, car_msg.red_2_y);
    int sz = sendto(sockfd, buffer, sizeof(buffer), 0, (sockaddr*)&addr, addr_len);
    //sendto(sockfd, "hello world", 11, 0, (sockaddr*)&addr, addr_len);
    if(sz >= 0)
    {        
        printf("Sended %d data to port: %d successful!\n", sz, portIn);
        return;
    } 
    printf("send message error!\n");
    return;
}

void _Socket_::closeSocket()
{
    close(sockfd);
}

void _Socket_::clearMsg()
{
        car_msg.blue_1_state = 0;
        car_msg.blue_2_state = 0;
        car_msg.red_1_state = 0;
        car_msg.red_2_state = 0;
        car_msg.blue_1_x = (double)4.04;
        car_msg.blue_1_y = (double)2.24;
        car_msg.blue_2_x = (double)4.04;
        car_msg.blue_2_y = (double)2.24;
        car_msg.red_1_x = (double)4.04;
        car_msg.red_1_y = (double)2.24;
        car_msg.red_2_x = (double)4.04;
        car_msg.red_2_y = (double)2.24;
        return;
}