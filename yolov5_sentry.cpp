#include <iostream>
#include <chrono>
#include <thread>
#include <mutex>
#include <stdlib.h>
#include <unistd.h>
#include "cuda_utils.h"
#include "logging.h"
#include "common.hpp"
#include "utils.h"
#include "calibrator.h"
#include "FlyCamera.h"
#include "car_classification.hpp"

#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE0 0  // each camera using one GPU
#define DEVICE1 1
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 1

// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

//Camera info
void getCameraIntrinsicMatrix(unsigned int camSN, cv::Matx33d &IntrinsicMatrix, cv::Vec4d &DistortionCoeffs)
{
    if(camSN == 17125973)
    {
        IntrinsicMatrix = {
                              1301.2777337637364781, -4.5144090274427802,   1070.1585987775554258,
                              0,                 1299.2267849954268968,     1030.0027897453960577, 
                              0,                 0,                     1
                                                                                      };
        DistortionCoeffs = {-0.3076151505860868, 0.0737292991186984, 0.0007870799241898, -0.0025294835070149};
    }
    else if(camSN == 17265336)
    {
        IntrinsicMatrix = {
                              1338.9754236997287080, -1.1824018557548868,   983.6721753175795584,
                              0,                 1338.4617539040229985,     1013.9923743317369826, 
                              0,                 0,                     1
                                                                                      };
        DistortionCoeffs = {-0.2988661964856638, 0.0684221548052301, -0.0010946689224421, 0.0003555977326127};
    }
    return;
}

static int get_width(int x, float gw, int divisor = 8) {
    //return math.ceil(x / divisor) * divisor
    if (int(x * gw) % divisor == 0) {
        return int(x * gw);
    }
    return (int(x * gw / divisor) + 1) * divisor;
}

static int get_depth(int x, float gd) {
    if (x == 1) {
        return 1;
    } else {
        return round(x * gd) > 1 ? round(x * gd) : 1;
    }
}

ICudaEngine* build_engine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, float& gd, float& gw, std::string& wts_name) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights(wts_name);

    /* ------ yolov5 backbone------ */
    auto focus0 = focus(network, weightMap, *data, 3, get_width(64, gw), 3, "model.0");
    auto conv1 = convBlock(network, weightMap, *focus0->getOutput(0), get_width(128, gw), 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = C3(network, weightMap, *conv1->getOutput(0), get_width(128, gw), get_width(128, gw), get_depth(3, gd), true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), get_width(256, gw), 3, 2, 1, "model.3");
    auto bottleneck_csp4 = C3(network, weightMap, *conv3->getOutput(0), get_width(256, gw), get_width(256, gw), get_depth(9, gd), true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), get_width(512, gw), 3, 2, 1, "model.5");
    auto bottleneck_csp6 = C3(network, weightMap, *conv5->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(9, gd), true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), get_width(1024, gw), 3, 2, 1, "model.7");
    auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), get_width(1024, gw), get_width(1024, gw), 5, 9, 13, "model.8");

    /* ------ yolov5 head ------ */
    auto bottleneck_csp9 = C3(network, weightMap, *spp8->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.9");
    auto conv10 = convBlock(network, weightMap, *bottleneck_csp9->getOutput(0), get_width(512, gw), 1, 1, 1, "model.10");

    auto upsample11 = network->addResize(*conv10->getOutput(0));
    assert(upsample11);
    upsample11->setResizeMode(ResizeMode::kNEAREST);
    upsample11->setOutputDimensions(bottleneck_csp6->getOutput(0)->getDimensions());

    ITensor* inputTensors12[] = { upsample11->getOutput(0), bottleneck_csp6->getOutput(0) };
    auto cat12 = network->addConcatenation(inputTensors12, 2);
    auto bottleneck_csp13 = C3(network, weightMap, *cat12->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.13");
    auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), get_width(256, gw), 1, 1, 1, "model.14");

    auto upsample15 = network->addResize(*conv14->getOutput(0));
    assert(upsample15);
    upsample15->setResizeMode(ResizeMode::kNEAREST);
    upsample15->setOutputDimensions(bottleneck_csp4->getOutput(0)->getDimensions());
	
    ITensor* inputTensors16[] = { upsample15->getOutput(0), bottleneck_csp4->getOutput(0) };
    auto cat16 = network->addConcatenation(inputTensors16, 2);

    auto bottleneck_csp17 = C3(network, weightMap, *cat16->getOutput(0), get_width(512, gw), get_width(256, gw), get_depth(3, gd), false, 1, 0.5, "model.17");

    // yolo layer 0
    IConvolutionLayer* det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);
    auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), get_width(256, gw), 3, 2, 1, "model.18");
    ITensor* inputTensors19[] = { conv18->getOutput(0), conv14->getOutput(0) };
    auto cat19 = network->addConcatenation(inputTensors19, 2);
    auto bottleneck_csp20 = C3(network, weightMap, *cat19->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.20");
    //yolo layer 1
    IConvolutionLayer* det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);
    auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), get_width(512, gw), 3, 2, 1, "model.21");
    ITensor* inputTensors22[] = { conv21->getOutput(0), conv10->getOutput(0) };
    auto cat22 = network->addConcatenation(inputTensors22, 2);
    auto bottleneck_csp23 = C3(network, weightMap, *cat22->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.23");
    IConvolutionLayer* det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

    auto yolo = addYoLoLayer(network, weightMap, det0, det1, det2);
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(4 * (1ULL << 30));  // 4GB
#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "./coco_calib/", "int8calib.table", INPUT_BLOB_NAME);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, float& gd, float& gw, std::string& wts_name) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = build_engine(maxBatchSize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}

void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, float& gd, float& gw) {
    if (argc < 3) return false;
    if (std::string(argv[1]) == "-s" && (argc == 5 || argc == 7)) {
        wts = std::string(argv[2]);
        engine = std::string(argv[3]);
        auto net = std::string(argv[4]);
        if (net == "s") {
            gd = 0.33;
            gw = 0.50;
        } else if (net == "m") {
            gd = 0.67;
            gw = 0.75;
        } else if (net == "l") {
            gd = 1.0;
            gw = 1.0;
        } else if (net == "x") {
            gd = 1.33;
            gw = 1.25;
        } else if (net == "c" && argc == 7) {
            gd = atof(argv[5]);
            gw = atof(argv[6]);
        } else {
            return false;
        }
    } else if (std::string(argv[1]) == "-d" && argc == 3) {
        engine = std::string(argv[2]);
    } else {
        return false;
    }
    return true;
}

#define PORT_IN_1 8080
#define PORT_IN_2 8080
#define SENTRY_ADDR_128 "192.168.1.128"
#define SENTRY_ADDR_105 "192.168.1.105"
FlyCamera cam;
std::mutex mtx, mtx2;
bool flag1 = false, flag2 = false;
_Socket_::CarMsg getMsg;

/*1号相机*/
CarState car_state1;
cv::Point2f mousePosition(0, 0), clickPosition[10];
cv::Point2f mapPosition[5];
int k = 0;
cv::Mat mask(3, 3, CV_64FC1);
cv::Mat org(3, 1, CV_64FC1);
cv::Mat warp_position(3, 1, CV_64FC1);

/*0号相机*/
CarState car_state2;
cv::Point2f mousePosition2(0, 0), clickPosition2[10];
cv::Point2f mapPosition2[5];
int k2 = 0;
cv::Mat mask2(3, 3, CV_64FC1);
cv::Mat org2(3, 1, CV_64FC1);
cv::Mat warp_position2(3, 1, CV_64FC1);

bool isEdge(cv::Point2f map[], cv::Point p, int camera_type)
{
    struct strightLine
    {
        double k, b;
    };
    strightLine ru, rd, lu, ld;
    if(camera_type == 0)
    {
        rd.k = ((double)(map[1].y - map[2].y) / (double)(map[1].x - map[2].x));
        rd.b = map[1].y - map[1].x * rd.k;
        ru.k = ((double)(map[1].y - map[3].y) / (double)(map[1].x - map[3].x));
        ru.b = map[1].y - map[1].x * ru.k;
        lu.k = ((double)(map[3].y - map[4].y) / (double)(map[3].x - map[4].x));
        lu.b = map[3].y - map[3].x * lu.k;
        ld.k = ((double)(map[4].y - map[2].y) / (double)(map[4].x - map[2].x));
        ld.b = map[2].y - map[2].x * ld.k;

        //for(int i=1;i<=4;i++) printf("%f %f %f %f %f %f %f %f\n", ru.k ,ru.b, rd.k, rd.b, lu.k, lu.b, ld.k, ld.b);
        if((p.y > p.x * ru.k + ru.b) && (p.y > p.x * lu.k + lu.b) && (p.y < p.x * rd.k + rd.b) && (p.y < p.x * ld.k + ld.b))
        {
            //printf("true\n");
            return true;
        }
        else
        {
            //printf("false\n");
            return false;
        }
    }
}

static void onMouse1(int event ,int x, int y, int flags, void* userInput)
{
    if(k >= 5) return;
    switch(event)
    {
        case(cv::EVENT_MOUSEMOVE):
        {
            mousePosition = cv::Point(x, y);
            break;
        }
        case(cv::EVENT_LBUTTONDOWN):
        { 
            k++;
            if(k <= 4)
            {                
                clickPosition[k] = cv::Point(x, y);
                mapPosition[k] = cv::Point(x, y);
            }
            if(k == 4) 
            {
                printf("请点击图像进行掩膜计算\n");
                break;
            }
            //printf("k = %d\n", k);
            break;
        }
    }
}

static void onMouse2(int event ,int x, int y, int flags, void* userInput)
{
    if(k2 >= 5) return;
    switch(event)
    {
        case(cv::EVENT_MOUSEMOVE):
        {
            mousePosition2 = cv::Point(x, y);
            break;
        }
        case(cv::EVENT_LBUTTONDOWN):
        { 
            k2++;
            if(k2 <= 4)
            {                
                clickPosition2[k2] = cv::Point(x, y);
                mapPosition2[k2] = cv::Point(x, y);
            }
            if(k2 == 4) 
            {
                printf("请点击图像进行掩膜计算\n");
                break;
            }
            //printf("k = %d\n", k);
            break;
        }
    }
}

void map1()
{
    cv::namedWindow("map1");
    cv::setMouseCallback("map1", onMouse1);
    k = 0;
    bool flag_map = false;
    while(1)
    {
        //std::cout<<k<<std::endl;
        cv::Mat img, dst;
        cam.camera_capture(1, img);
        if (img.empty())
        {
            printf("capture img fail!\n");
            continue;
        }

        cv::Matx33d intrinsic_matrix;
        cv::Vec4d distortion_coeffs;
        getCameraIntrinsicMatrix(cam.cam_serial_num[1], intrinsic_matrix, distortion_coeffs);
        cv::Mat undistort_img; //图像畸变校正（耗时）

        cv::undistort(img, undistort_img, intrinsic_matrix, distortion_coeffs,
            cv::getOptimalNewCameraMatrix(intrinsic_matrix, distortion_coeffs, img.size(), 1, img.size(), 0));

        // cv::Mat dst = img.clone();
        cv::resize(undistort_img,dst,cv::Size(1024,1024));

        int sk = k <= 4 ? k : 4;
        for(int i=1;i<=sk;i++)
        {
            cv::circle(dst, clickPosition[i], 3, cv::Scalar(0, 255, 0), -1, 8);
        }
        for(int i=1;i<=sk;i++)
        {
            for(int j=i+1;j<=sk;j++)
            {
                cv::line(dst, clickPosition[i], clickPosition[j], cv::Scalar(0, 0, 255), 1, 4);
            }
        }
        if(k < 4)
        {            
            cv::circle(dst, mousePosition, 3, cv::Scalar(0, 255, 0), -1, 8);
            for(int i=1;i<=sk;i++)
            {
                cv::line(dst, clickPosition[i], mousePosition, cv::Scalar(0, 0, 255), 1, 4);
            }
        }         
        if(k == 5 && !flag_map)
        {            
            cv::Point2f camera_view[] = {clickPosition[1], clickPosition[2], clickPosition[3], clickPosition[4]};
            cv::Point2f god_view[] = {cv::Point2f(808, 0), cv::Point2f(808, 448), cv::Point2f(0, 0), cv::Point2f(0, 448)};
            mask = cv::getPerspectiveTransform(camera_view, god_view);
            std::cout<<"地图1标定完成，按Q打开识别模块。。。"<<std::endl;
            flag_map = true;
        }

        cv::imshow("map1", dst);    

        char c = cv::waitKey(1);
        if(c == 'q')
        {
            cv::destroyWindow("map1");
            break;
        }
    }
}

void map2()
{
    cv::namedWindow("map2");
    cv::setMouseCallback("map2", onMouse2);
    k = 0;
    bool flag_map = false;
    while(1)
    {
        //std::cout<<k<<std::endl;
        cv::Mat img, dst;
        cam.camera_capture(0, img);
        if (img.empty())
        {
            printf("capture img fail!\n");
            continue;
        }

        cv::Matx33d intrinsic_matrix;
        cv::Vec4d distortion_coeffs;
        getCameraIntrinsicMatrix(cam.cam_serial_num[0], intrinsic_matrix, distortion_coeffs);
        cv::Mat undistort_img; //图像畸变校正（耗时）

        cv::undistort(img, undistort_img, intrinsic_matrix, distortion_coeffs,
            cv::getOptimalNewCameraMatrix(intrinsic_matrix, distortion_coeffs, img.size(), 1, img.size(), 0));

        // cv::Mat dst = img.clone();
        cv::resize(undistort_img,dst,cv::Size(1024,1024));

        int sk2 = k2 <= 4 ? k2 : 4;
        for(int i=1;i<=sk2;i++)
        {
            cv::circle(dst, clickPosition2[i], 3, cv::Scalar(0, 255, 0), -1, 8);
        }
        for(int i=1;i<=sk2;i++)
        {
            for(int j=i+1;j<=sk2;j++)
            {
                cv::line(dst, clickPosition2[i], clickPosition2[j], cv::Scalar(0, 0, 255), 1, 4);
            }
        }
        if(k2 < 4)
        {            
            cv::circle(dst, mousePosition2, 3, cv::Scalar(0, 255, 0), -1, 8);
            for(int i=1;i<=sk2;i++)
            {
                cv::line(dst, clickPosition2[i], mousePosition2, cv::Scalar(0, 0, 255), 1, 4);
            }
        }         
        if(k2 == 5 && !flag_map)
        {            
            cv::Point2f camera_view[] = {clickPosition2[1], clickPosition2[2], clickPosition2[3], clickPosition2[4]};
            cv::Point2f god_view[] = {cv::Point2f(808, 0), cv::Point2f(808, 448), cv::Point2f(0, 0), cv::Point2f(0, 448)};
            mask2 = cv::getPerspectiveTransform(camera_view, god_view);
            std::cout<<"地图2标定完成，按Q打开识别模块。。。"<<std::endl;
            flag_map = true;
        }

        cv::imshow("map2", dst);    

        char c = cv::waitKey(1);
        if(c == 'q')
        {
            cv::destroyWindow("map2");
            break;
        }
    }
}

void detect1(int camSN, std::string engine_name)
{
    cudaSetDevice(DEVICE0);
    // deserialize the .engine and run inference
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        return;
    }
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    // prepare input data ---------------------------
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
    //    data[i] = 1.0;
    static float prob[BATCH_SIZE * OUTPUT_SIZE];
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    void* buffers[2];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cv::Matx33d intrinsic_matrix;
    cv::Vec4d distortion_coeffs;
    getCameraIntrinsicMatrix(camSN, intrinsic_matrix, distortion_coeffs);

    while(1)
    {
        auto start = std::chrono::system_clock::now();

        //std::cout<<"*************************"<<std::endl;

        //printf("相机SN号：%u\n", camSN);

        cv::Mat img;
        cam.camera_capture(1, img);
        if (img.empty())
        {
            printf("capture img fail!\n");
            continue;
        }

        cv::Mat undistort_img; //图像畸变校正（耗时）

        cv::undistort(img, undistort_img, intrinsic_matrix, distortion_coeffs,
            cv::getOptimalNewCameraMatrix(intrinsic_matrix, distortion_coeffs, img.size(), 1, img.size(), 0));

        cv::Mat dst = undistort_img.clone();
        //cv::Mat dst = img.clone();
        cv::resize(dst,dst,cv::Size(1024,1024));

        cv::Mat pr_img = preprocess_img(dst, INPUT_W, INPUT_H); // letterbox BGR to RGB
        int i = 0;
        for (int row = 0; row < INPUT_H; ++row) {
            uchar* uc_pixel = pr_img.data + row * pr_img.step;
            for (int col = 0; col < INPUT_W; ++col) {
                data[0 * 3 * INPUT_H * INPUT_W + i] = (float)uc_pixel[2] / 255.0;
                data[0 * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
                data[0 * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
                uc_pixel += 3;
                ++i;
            }
        }

        // Run inference
        doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
        std::vector<std::vector<Yolo::Detection>> batch_res(1);

        auto& res = batch_res[0];
        nms(res, &prob[0], CONF_THRESH, NMS_THRESH);

        cv::Mat warp_img;
        cv::warpPerspective(dst, warp_img, mask, cv::Size(808, 448));

        //std::cout << "识别目标个数： " << res.size() << std::endl;
        for (size_t j = 0; j < res.size(); j++) 
        {
            cv::Rect r = get_rect(dst, res[j].bbox);
            double centerX = (r.tl().x + r.br().x) / 2;
            double centerY = r.br().y - (r.br().y - r.tl().y) / 4;

            if(!isEdge(clickPosition, cv::Point(centerX, centerY), 0)) continue;

            org = (cv::Mat_<double>(3, 1)<<centerX, centerY, 1);
            warp_position = mask * org;
            centerX = (*warp_position.ptr<double>(0, 0)) / (*warp_position.ptr<double>(2, 0));
            centerY = (*warp_position.ptr<double>(1, 0)) / (*warp_position.ptr<double>(2, 0));

            cv::circle(warp_img, cv::Point(centerX, centerY), 8, cv::Scalar(0, 0, 255), -1, 8);
            mtx.lock();
            car_state1.CarClassify((int)res[j].class_id, centerX, centerY);
            mtx.unlock();

            cv::rectangle(dst, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
            cv::putText(dst, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);

        }

        mtx.lock();
        flag1 = true;
        mtx.unlock();

        cv::imshow("image", dst);
        cv::imshow("warp_image", warp_img);

        auto end = std::chrono::system_clock::now();
        //std::cout <<"帧处理总时间： "<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        cv::waitKey(1);
    }

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
}

void detect2(int camSN, std::string engine_name)
{
    cudaSetDevice(DEVICE1);
    // deserialize the .engine and run inference
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        return;
    }
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    // prepare input data ---------------------------
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
    //    data[i] = 1.0;
    static float prob[BATCH_SIZE * OUTPUT_SIZE];
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    void* buffers[2];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cv::Matx33d intrinsic_matrix;
    cv::Vec4d distortion_coeffs;
    getCameraIntrinsicMatrix(camSN, intrinsic_matrix, distortion_coeffs);

    while(1)
    {
        auto start = std::chrono::system_clock::now();

        //std::cout<<"*************************"<<std::endl;

        //printf("相机SN号：%u\n", camSN);

        cv::Mat img2;
        cam.camera_capture2(0, img2);
        if (img2.empty())
        {
            printf("capture img fail!\n");
            continue;
        }
        cv::Mat undistort_img2; //图像畸变校正（耗时）

        cv::undistort(img2, undistort_img2, intrinsic_matrix, distortion_coeffs,
            cv::getOptimalNewCameraMatrix(intrinsic_matrix, distortion_coeffs, img2.size(), 1, img2.size(), 0));

        cv::Mat dst2 = undistort_img2.clone();
        //cv::Mat dst = img.clone();
        cv::resize(dst2,dst2,cv::Size(1024,1024));
        cv::Mat pr_img2 = preprocess_img(dst2, INPUT_W, INPUT_H); // letterbox BGR to RGB

        int i = 0;
        for (int row = 0; row < INPUT_H; ++row) {
            uchar* uc_pixel = pr_img2.data + row * pr_img2.step;
            for (int col = 0; col < INPUT_W; ++col) {
                data[0 * 3 * INPUT_H * INPUT_W + i] = (float)uc_pixel[2] / 255.0;
                data[0 * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
                data[0 * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
                uc_pixel += 3;
                ++i;
            }
        }

        // Run inference
        doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
        std::vector<std::vector<Yolo::Detection>> batch_res(1);

        auto& res = batch_res[0];
        nms(res, &prob[0], CONF_THRESH, NMS_THRESH);

        cv::Mat warp_img2;
        cv::warpPerspective(dst2, warp_img2, mask2, cv::Size(808, 448));

        //std::cout << "识别目标个数： " << res.size() << std::endl;
        for (size_t j = 0; j < res.size(); j++) 
        {
            cv::Rect r = get_rect(dst2, res[j].bbox);
            double centerX = (r.tl().x + r.br().x) / 2;
            double centerY = r.br().y - (r.br().y - r.tl().y) / 4;

            if(!isEdge(clickPosition2, cv::Point(centerX, centerY), 0)) continue;

            org2 = (cv::Mat_<double>(3, 1)<<centerX, centerY, 1);
            warp_position2 = mask2 * org2;
            centerX = (*warp_position2.ptr<double>(0, 0)) / (*warp_position2.ptr<double>(2, 0));
            centerY = (*warp_position2.ptr<double>(1, 0)) / (*warp_position2.ptr<double>(2, 0));

            cv::circle(warp_img2, cv::Point(centerX, centerY), 8, cv::Scalar(0, 0, 255), -1, 8);

            mtx2.lock();
            if(cam.cam_serial_num[0] == 17265336)
            {
                centerX = 808 - centerX;
                centerY = 448 - centerY;
            }     
            car_state2.CarClassify((int)res[j].class_id, centerX, centerY);
            mtx2.unlock();

            cv::rectangle(dst2, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
            cv::putText(dst2, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);

        }

        mtx2.lock();
        flag2 = true;
        mtx2.unlock();
        
        cv::imshow("image2", dst2);
        cv::imshow("warp_image2", warp_img2);

        auto end = std::chrono::system_clock::now();
        //std::cout <<"帧处理总时间： "<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        cv::waitKey(1);
    }

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
}

Position pos_process(Position p1, Position p2)
{
    Position ans;
    if((p1.state == 0) && (p2.state == 1))
    {
        ans.poi.x = p2.poi.x;
        ans.poi.y = p2.poi.y;
        ans.state = 1;
    }
    else if((p1.state == 1) && (p2.state == 0))
    {
        ans.poi.x = p1.poi.x;
        ans.poi.y = p1.poi.y;
        ans.state = 1;
    }
    else if((p1.state == 0) && (p2.state == 0))
    {
        ans.poi.x = (double)4.04;
        ans.poi.y = (double)2.24;
        ans.state = 0;
    }
    else
    {
        ans.poi.x = (double)((p1.poi.x + p2.poi.x) / 2.0);
        ans.poi.y = (double)((p1.poi.y + p2.poi.y) / 2.0);
        ans.state = 1;
    }
    return ans;
}

void _clearMsg()
{
    getMsg.blue_1_state = false;
    getMsg.blue_1_x = (double)4.04;
    getMsg.blue_1_y = (double)2.24;
    getMsg.blue_2_state = false;
    getMsg.blue_2_x = (double)4.04;
    getMsg.blue_2_y = (double)2.24;
    getMsg.red_1_state = false;
    getMsg.red_1_x = (double)4.04;
    getMsg.red_1_y = (double)2.24;
    getMsg.red_2_state = false;
    getMsg.red_2_x = (double)4.04;
    getMsg.red_2_y = (double)2.24;
    return;
}

void _sendMsg()
{
    _Socket_ skt1, skt2;
    skt1.setSocket(SENTRY_ADDR_128, PORT_IN_1);
    skt2.setSocket(SENTRY_ADDR_105, PORT_IN_2);
    auto time_ = std::chrono::system_clock::now(), last_time_ = std::chrono::system_clock::now();
    while(true)
    {
        if(cam.camera_num == 2)
        {
            mtx.lock();mtx2.lock();
            if(flag1 == true && flag2 == true)
            {
                printf("#########################\n");
                printf("即将发送的位置信息：\n");
                Position blue1 = pos_process(car_state1.blue.one, car_state2.blue.one);
                getMsg.blue_1_state = blue1.state;
                getMsg.blue_1_x = blue1.poi.x;
                getMsg.blue_1_y = blue1.poi.y;
                printf("蓝色一号车:  state: %d  X:%f, Y: %f\n", getMsg.blue_1_state, getMsg.blue_1_x, getMsg.blue_1_y);
                Position blue2 = pos_process(car_state1.blue.two, car_state2.blue.two);
                getMsg.blue_2_state = blue2.state;
                getMsg.blue_2_x = blue2.poi.x;
                getMsg.blue_2_y = blue2.poi.y;
                printf("蓝色二号车:  state: %d  X:%f, Y: %f\n", getMsg.blue_2_state, getMsg.blue_2_x, getMsg.blue_2_y);
                Position red1 = pos_process(car_state1.red.one, car_state2.red.one);
                getMsg.red_1_state = red1.state;
                getMsg.red_1_x = red1.poi.x;
                getMsg.red_1_y = red1.poi.y;
                printf("红色一号车:  state: %d  X:%f, Y: %f\n", getMsg.red_1_state, getMsg.red_1_x, getMsg.red_1_y);
                Position red2 = pos_process(car_state1.red.two, car_state2.red.two);
                getMsg.red_2_state = red2.state;
                getMsg.red_2_x = red2.poi.x;
                getMsg.red_2_y = red2.poi.y;
                printf("红色二号车:  state: %d  X:%f, Y: %f\n", getMsg.red_2_state, getMsg.red_2_x, getMsg.red_2_y);
                skt1.car_msg = getMsg;
                skt2.car_msg = getMsg;

                skt1.sendSocket();
                skt1.clearMsg();
                skt2.sendSocket();
                skt2.clearMsg();
                _clearMsg();
                car_state1.clearCarState();
                car_state2.clearCarState();

                flag1 = false;flag2 = false;

                time_ = std::chrono::system_clock::now();
                //std::cout <<"帧处理总时间： "<< std::chrono::duration_cast<std::chrono::milliseconds>(time_ - last_time_).count() << "ms" << std::endl;
                last_time_ = time_;
            }
            mtx.unlock();mtx2.unlock();
        }
        else if(cam.camera_num == 1)
        {
            mtx2.lock();
            if(flag2 == true)
            {
                skt1.IntegrateInfo(car_state2);
                skt1.sendSocket();
                skt1.clearMsg();
                skt2.IntegrateInfo(car_state2);
                skt2.sendSocket();
                skt2.clearMsg();
                car_state2.clearCarState();
                flag2 = false;
            }
            mtx2.unlock();
        }
    }
    skt1.closeSocket();
    skt2.closeSocket();
}

int main(int argc, char** argv) {

    std::string wts_name = "";
    std::string engine_name = "";
    float gd = 0.0f, gw = 0.0f;
    if (!parse_args(argc, argv, wts_name, engine_name, gd, gw)) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./yolov5 -s [.wts] [.engine] [s/m/l/x or c gd gw]  // serialize model to plan file" << std::endl;
        std::cerr << "./yolov5 -d [.engine] // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    if (!wts_name.empty()) {
        IHostMemory* modelStream{ nullptr };
        APIToModel(BATCH_SIZE, &modelStream, gd, gw, wts_name);
        assert(modelStream != nullptr);
        std::ofstream p(engine_name, std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    }

    if (!cam.camera_init())
    {
        printf("camera init error!\n");
        return -1;
    }

    map2();
    if(cam.camera_num == 2) map1();


    //while(1);
    std::thread camera1(detect2, cam.cam_serial_num[0], engine_name);
    sleep(3);
    std::thread sendsocket(_sendMsg);
    sleep(3);

    if(cam.camera_num == 2)
    {        
        std::thread camera2(detect1, cam.cam_serial_num[1], engine_name);
        sleep(3);
        camera2.join();
    }

    camera1.join();
    sendsocket.join();

    // Print histogram of the output distribution
    //std::cout << "\nOutput:\n\n";
    //for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
    //{
    //    std::cout << prob[i] << ", ";
    //    if (i % 10 == 0) std::cout << std::endl;
    //}
    //std::cout << std::endl;

    return 0;
}
