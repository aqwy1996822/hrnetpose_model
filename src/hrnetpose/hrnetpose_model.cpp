#include "hrnetpose/hrnetpose_model.h"

Hrnetpose_model::Hrnetpose_model()
{
    cudaSetDevice(DEVICE);
}
Hrnetpose_model::~Hrnetpose_model()
{
    if (predict_flag)
    {
        // Release stream and buffers
        cudaStreamDestroy(stream);
        CUDA_CHECK(cudaFree(buffers[inputIndex]));
        CUDA_CHECK(cudaFree(buffers[outputIndex]));
        // Destroy the engine
        context->destroy();
        engine->destroy();
        runtime->destroy();
    }

}
void Hrnetpose_model::doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueueV2(buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}
void Hrnetpose_model::get_realpoint(cv::Rect rect, std::vector<cv::Point> &hint_poses) {
    int l, t;
    float r_w = OUTPUT_W / (rect.width * 1.0);
    float r_h = OUTPUT_H / (rect.height * 1.0);


    if (r_h > r_w) {
        for (int i = 0; i < hint_poses.size(); ++i) {
            l = hint_poses[i].x;
            t = hint_poses[i].y - (OUTPUT_H - r_w * rect.height) / 2;
            hint_poses[i].x = l / r_w;
            hint_poses[i].y = t / r_w;
        }
    }
    else {
        for (int i = 0; i < hint_poses.size(); ++i) {
            l = hint_poses[i].x  - (OUTPUT_W - r_h * rect.width) / 2;
            t = hint_poses[i].y ;
            hint_poses[i].x = l / r_h;
            hint_poses[i].y = t / r_h;
        }
    }
}
// letterbox; BGR to RGB
cv::Mat Hrnetpose_model::preprocess_img(cv::Mat img, int input_w, int input_h) {
    int w, h, x, y;
    float r_w = input_w / (img.cols*1.0);
    float r_h = input_h / (img.rows*1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    } else {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128)); //??????
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}

void Hrnetpose_model::deserialize_engine(std::string engine_name)
{
//    std::string engine_name = "../hrnet32_256x192_dynamic.engine";

    // deserialize the .engine and run inference
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
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
    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    assert(engine->getNbBindings() == 2);
    inputIndex = engine->getBindingIndex("input");
    outputIndex = engine->getBindingIndex("output");

    assert(inputIndex == 0);
    assert(outputIndex == 1);

    // Create stream
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void**)&buffers[inputIndex], MAXBATCHSIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&buffers[outputIndex], MAXBATCHSIZE * OUTPUT_SIZE * sizeof(float)));

    predict_flag=true;
}
void Hrnetpose_model::serialize_engine()
{
    build_engine();
}

void Hrnetpose_model::build_engine()
{
    auto builder = nvinfer1::createInferBuilder(gLogger);
//    auto network = builder->createNetworkV2();
    auto network = builder->createNetworkV2(1U<<static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));


    auto parser = nvonnxparser::createParser(*network, gLogger);
    parser->parseFromFile("../hrnet32_b-1_256x192.onnx", int(nvinfer1::ILogger::Severity::kWARNING));

    auto config = builder->createBuilderConfig();

    config->setMaxWorkspaceSize(1 << 30);


    Dims mInputDims = network->getInput(0)->getDimensions();

    Dims mOutputDims = network->getOutput(0)->getDimensions();

    auto profile = builder->createOptimizationProfile();

    profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMIN, Dims4{1, mInputDims.d[1], mInputDims.d[2], mInputDims.d[3]});
    profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kOPT, Dims4{MAXBATCHSIZE, mInputDims.d[1], mInputDims.d[2], mInputDims.d[3]});
    profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMAX, Dims4{MAXBATCHSIZE, mInputDims.d[1], mInputDims.d[2], mInputDims.d[3]});

    config->addOptimizationProfile(profile);

    // Create a calibration profile.
    auto profileCalib = builder->createOptimizationProfile();

    // We do not need to check the return of setDimension and setCalibrationProfile here as all dims are explicitly set
    profileCalib->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMIN, Dims4{1, mInputDims.d[1], mInputDims.d[2], mInputDims.d[3]});
    profileCalib->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kOPT, Dims4{1, mInputDims.d[1], mInputDims.d[2], mInputDims.d[3]});
    profileCalib->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMAX, Dims4{1, mInputDims.d[1], mInputDims.d[2], mInputDims.d[3]});
    config->setCalibrationProfile(profileCalib);

    if (int8_flag)
    {
        std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
        assert(builder->platformHasFastInt8());
        config->setFlag(BuilderFlag::kINT8);
        Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "../coco_calib/", "int8calib.table", "input");
        config->setInt8Calibrator(calibrator);
    }
    else
    {
        config->setFlag(BuilderFlag::kFP16);
    }

    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    // serialize
    IHostMemory *seriallizedModel = engine->serialize();
    // save engine
    if (int8_flag)
    {
        std::ofstream p("../hrnet32_256x192_dynamic_int8.engine", std::ios::binary);
        if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
        }
        p.write(reinterpret_cast<const char*>(seriallizedModel->data()), seriallizedModel->size());
    }
    else
    {
        std::ofstream p("../hrnet32_256x192_dynamic.engine", std::ios::binary);
        if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
        }
        p.write(reinterpret_cast<const char*>(seriallizedModel->data()), seriallizedModel->size());
    }


}
void Hrnetpose_model::runtest2()
{

    std::string engine_name = "../hrnet32_256x192_dynamic.engine";

    // deserialize the .engine and run inference
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
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



    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    assert(engine->getNbBindings() == 2);
    inputIndex = engine->getBindingIndex("input");
    outputIndex = engine->getBindingIndex("output");

    assert(inputIndex == 0);
    assert(outputIndex == 1);

    // Create stream
    CUDA_CHECK(cudaStreamCreate(&stream));

    cv::VideoCapture cap;
    cv::Mat img;

    cap.open(0);
//    std::vector<cv::Rect> rects={cv::Rect(0, 0, 320, 240), cv::Rect(320, 0, 320, 240), cv::Rect(0, 240, 320, 240), cv::Rect(320, 240, 320, 240)};
//    std::vector<cv::Rect> rects={cv::Rect(0, 0, 320, 480), cv::Rect(320, 0, 320, 480)};
    std::vector<cv::Rect> rects={cv::Rect(0, 0, 640, 480)};

    const int fcount = rects.size();



    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void**)&buffers[inputIndex], MAXBATCHSIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&buffers[outputIndex], MAXBATCHSIZE * OUTPUT_SIZE * sizeof(float)));

//    context->setOptimizationProfile(1);
    context->setBindingDimensions(0, Dims4{fcount, 3, 256, 192});
    while (cap.isOpened()){
        cap>>img;

        std::vector<std::vector<cv::Point>> people_hint_poses;
        std::vector<std::vector<float>> people_points_conf;
        // batch_size=1
        auto start01 = std::chrono::system_clock::now();


        for (int b = 0; b < fcount; b++) {
            cv::Mat pr_img = preprocess_img(img(rects[b]), INPUT_W, INPUT_H); // letterbox BGR to RGB
            int i = b * INPUT_H * INPUT_W * 3;
            for (int row = 0; row < INPUT_H; ++row) {
                uchar *uc_pixel = pr_img.data + row * pr_img.step;
                for (int col = 0; col < INPUT_W; ++col) {
                    input[i] = ((float) uc_pixel[2] / 255.0 - 0.485) / 0.229;
                    input[i + INPUT_H * INPUT_W] = ((float) uc_pixel[1] / 255.0 - 0.456) / 0.224;
                    input[i + 2 * INPUT_H * INPUT_W] = ((float) uc_pixel[0] / 255.0 - 0.406) / 0.225;
                    uc_pixel += 3;
                    ++i;
                }
            }
        }
        auto end01 = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end01 - start01).count() << "ms" << std::endl;


        auto start02 = std::chrono::system_clock::now();
        doInference(*context, stream, (void**)buffers, input, output, fcount);
        auto end02 = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end02 - start02).count() << "ms" << std::endl;


        auto start03 = std::chrono::system_clock::now();
        for (int b = 0; b < fcount; b++) {
            std::vector<cv::Point> hint_poses;
            std::vector<float> points_conf;
            int outb_i=b*OUTPUT_SIZE;//output_batch_index
            for (int part_i = 0; part_i < PART_NUM; part_i++) {

                float* ptr=outb_i+output+part_i*OUTPUT_H*OUTPUT_W;
                int maxpos = std::max_element(ptr, ptr+OUTPUT_H*OUTPUT_W)-ptr;

                points_conf.push_back(ptr[maxpos]);
                hint_poses.push_back(cv::Point(maxpos%OUTPUT_W,maxpos/OUTPUT_W));
            }
            points_conf.push_back((points_conf[5]+points_conf[6])/2.0);
            hint_poses.push_back(cv::Point((hint_poses[5].x+hint_poses[6].x)/2,(hint_poses[5].y+hint_poses[6].y)/2));

            get_realpoint(rects[b], hint_poses);
            people_hint_poses.push_back(hint_poses);
            people_points_conf.push_back(points_conf);
        }
        auto end03 = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end03 - start03).count() << "ms" << std::endl;

        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end03 - start01).count() << "ms" << std::endl;
        //���ӻ�
        for (int b = 0; b < fcount; b++) {
            for (int part_i = 0; part_i < PART_NUM+1; part_i++) {
                if (people_points_conf[b][part_i]>CONF_THRESH){
                    cv::circle(img(rects[b]), people_hint_poses[b][part_i],5, cv::Scalar(255,255,255),-1,0);
                }
            }
            for (int line_i = 0; line_i < PART_LINE.size(); line_i++) {
                if (people_points_conf[b][PART_LINE[line_i][0]]>CONF_THRESH && people_points_conf[b][PART_LINE[line_i][1]]>CONF_THRESH){
                    cv::line(img(rects[b]), people_hint_poses[b][PART_LINE[line_i][0]], people_hint_poses[b][PART_LINE[line_i][1]], colors[line_i] , 3, 0);
                }
            }
        }
        cv::imshow("result",img);
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
void Hrnetpose_model::run_frame()
{
    cv::VideoCapture cap;
    cv::Mat img;

    cap.open(0);
//    std::vector<cv::Rect> rects={cv::Rect(0, 0, 320, 240), cv::Rect(320, 0, 320, 240), cv::Rect(0, 240, 320, 240), cv::Rect(320, 240, 320, 240)};
    std::vector<cv::Rect> rects={cv::Rect(0, 0, 320, 480), cv::Rect(320, 0, 320, 480)};
//    std::vector<cv::Rect> rects={cv::Rect(0, 0, 640, 480)};

    const int fcount = rects.size();

//    context->setOptimizationProfile(1);
    context->setBindingDimensions(0, Dims4{fcount, 3, 256, 192});
    while (cap.isOpened()){
        cap>>img;

        std::vector<std::vector<cv::Point>> people_hint_poses;
        std::vector<std::vector<float>> people_points_conf;
        // batch_size=1
        auto start01 = std::chrono::system_clock::now();


        for (int b = 0; b < fcount; b++) {
            cv::Mat pr_img = preprocess_img(img(rects[b]), INPUT_W, INPUT_H); // letterbox BGR to RGB
            int i = b * INPUT_H * INPUT_W * 3;
            for (int row = 0; row < INPUT_H; ++row) {
                uchar *uc_pixel = pr_img.data + row * pr_img.step;
                for (int col = 0; col < INPUT_W; ++col) {
                    input[i] = ((float) uc_pixel[2] / 255.0 - 0.485) / 0.229;
                    input[i + INPUT_H * INPUT_W] = ((float) uc_pixel[1] / 255.0 - 0.456) / 0.224;
                    input[i + 2 * INPUT_H * INPUT_W] = ((float) uc_pixel[0] / 255.0 - 0.406) / 0.225;
                    uc_pixel += 3;
                    ++i;
                }
            }
        }
        auto end01 = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end01 - start01).count() << "ms" << std::endl;


        auto start02 = std::chrono::system_clock::now();
        doInference(*context, stream, (void**)buffers, input, output, fcount);
        auto end02 = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end02 - start02).count() << "ms" << std::endl;


        auto start03 = std::chrono::system_clock::now();
        for (int b = 0; b < fcount; b++) {
            std::vector<cv::Point> hint_poses;
            std::vector<float> points_conf;
            int outb_i=b*OUTPUT_SIZE;//output_batch_index
            for (int part_i = 0; part_i < PART_NUM; part_i++) {

                float* ptr=outb_i+output+part_i*OUTPUT_H*OUTPUT_W;
                int maxpos = std::max_element(ptr, ptr+OUTPUT_H*OUTPUT_W)-ptr;

                points_conf.push_back(ptr[maxpos]);
                hint_poses.push_back(cv::Point(maxpos%OUTPUT_W,maxpos/OUTPUT_W));
            }
            points_conf.push_back((points_conf[5]+points_conf[6])/2.0);
            hint_poses.push_back(cv::Point((hint_poses[5].x+hint_poses[6].x)/2,(hint_poses[5].y+hint_poses[6].y)/2));

            get_realpoint(rects[b], hint_poses);
            people_hint_poses.push_back(hint_poses);
            people_points_conf.push_back(points_conf);
        }
        auto end03 = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end03 - start03).count() << "ms" << std::endl;

        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end03 - start01).count() << "ms"<< std::endl;
        for (int b = 0; b < fcount; b++) {
            for (int part_i = 0; part_i < PART_NUM+1; part_i++) {
                if (people_points_conf[b][part_i]>CONF_THRESH){
                    cv::circle(img(rects[b]), people_hint_poses[b][part_i],5, cv::Scalar(255,255,255),-1,0);
                }
            }
            for (int line_i = 0; line_i < PART_LINE.size(); line_i++) {
                if (people_points_conf[b][PART_LINE[line_i][0]]>CONF_THRESH && people_points_conf[b][PART_LINE[line_i][1]]>CONF_THRESH){
                    cv::line(img(rects[b]), people_hint_poses[b][PART_LINE[line_i][0]], people_hint_poses[b][PART_LINE[line_i][1]], colors[line_i] , 3, 0);
                }
            }
        }
        cv::imshow("result",img);
        cv::waitKey(1);

    }
}