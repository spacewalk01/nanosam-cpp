#include "trt_module.h"
#include "logging.h"
#include "cuda_utils.h"
#include "config.h"
#include "macros.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

static Logger gLogger;


std::string getFileExtension(const std::string& filePath) {
    size_t dotPos = filePath.find_last_of(".");
    if (dotPos != std::string::npos) {
        return filePath.substr(dotPos + 1);
    }
    return ""; // No extension found
}

TRTModule::TRTModule(string modelPath, vector<string> inputNames, vector<string> outputNames, bool isDynamicShape, bool isFP16)
{
    if (getFileExtension(modelPath) == "onnx")
    {
        cout << "Building Engine from " << modelPath << endl;
        build(modelPath, inputNames, outputNames, isDynamicShape, isFP16);
    }
    else
    {
        cout << "Deserializing Engine." << endl;
        deserializeEngine(modelPath, inputNames, outputNames);
    }
}

TRTModule::~TRTModule()
{
    // Release stream and buffers
    cudaStreamDestroy(mCudaStream);
    for (int i = 0; i < mGpuBuffers.size(); i++)
        CUDA_CHECK(cudaFree(mGpuBuffers[i]));
    for (int i = 0; i < mCpuBuffers.size(); i++)    
        delete[] mCpuBuffers[i];
    
    // Destroy the engine
    delete mContext;
    delete mEngine;
    delete mRuntime;
}

void TRTModule::build(string onnxPath, vector<string> inputNames, vector<string> outputNames, bool isDynamicShape, bool isFP16)
{
    auto builder = createInferBuilder(gLogger);
    assert(builder != nullptr);

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    assert(network != nullptr);

    IBuilderConfig* config = builder->createBuilderConfig();
    assert(config != nullptr);

    if (isDynamicShape) // Only designed for NanoSAM mask decoder
    {
        auto profile = builder->createOptimizationProfile();

        profile->setDimensions(inputNames[1].c_str(), OptProfileSelector::kMIN, Dims3{ 1, 1, 2 });
        profile->setDimensions(inputNames[1].c_str(), OptProfileSelector::kOPT, Dims3{ 1, 1, 2 });
        profile->setDimensions(inputNames[1].c_str(), OptProfileSelector::kMAX, Dims3{ 1, 10, 2 });

        profile->setDimensions(inputNames[2].c_str(), OptProfileSelector::kMIN, Dims2{ 1, 1 });
        profile->setDimensions(inputNames[2].c_str(), OptProfileSelector::kOPT, Dims2{ 1, 1 });
        profile->setDimensions(inputNames[2].c_str(), OptProfileSelector::kMAX, Dims2{ 1, 10 });

        config->addOptimizationProfile(profile);
    }

    if (isFP16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }

    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
    assert(parser != nullptr);

    bool parsed = parser->parseFromFile(onnxPath.c_str(), static_cast<int>(gLogger.getReportableSeverity()));
    assert(parsed != nullptrt);


    // CUDA stream used for profiling by the builder.
    assert(mCudaStream != nullptr);

    IHostMemory* plan{ builder->buildSerializedNetwork(*network, *config) };
    assert(plan != nullptr);

    mRuntime = createInferRuntime(gLogger);
    assert(mRuntime != nullptr);

    mEngine = mRuntime->deserializeCudaEngine(plan->data(), plan->size(), nullptr);
    assert(mEngine != nullptr);

    mContext = mEngine->createExecutionContext();
    assert(mContext != nullptr);

    delete network;
    delete config;
    delete parser;
    delete plan;

    initialize(inputNames, outputNames);
}

void TRTModule::deserializeEngine(string engine_name, vector<string> inputNames, vector<string> outputNames)
{
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        assert(false);
    }
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char* serializedEngine = new char[size];
    assert(serializedEngine);
    file.read(serializedEngine, size);
    file.close();

    mRuntime = createInferRuntime(gLogger);
    assert(mRuntime);
    mEngine = mRuntime->deserializeCudaEngine(serializedEngine, size);
    assert(*mEngine);
    mContext = mEngine->createExecutionContext();
    assert(*mContext);
    delete[] serializedEngine;

    assert(mEngine->getNbBindings() != inputNames.size() + outputNames.size()); 

    initialize(inputNames, outputNames);
}

void TRTModule::initialize(vector<string> inputNames, vector<string> outputNames)
{
    for (int i = 0; i < inputNames.size(); i++)
    {
        const int inputIndex = mEngine->getBindingIndex(inputNames[i].c_str());
    }

    for (int i = 0; i < outputNames.size(); i++)
    {
        const int outputIndex = mEngine->getBindingIndex(outputNames[i].c_str());
    }

    mGpuBuffers.resize(mEngine->getNbBindings());
    mCpuBuffers.resize(mEngine->getNbBindings());

    for (size_t i = 0; i < mEngine->getNbBindings(); ++i)
    {
        size_t binding_size = getSizeByDim(mEngine->getBindingDimensions(i));
        mBufferBindingSizes.push_back(binding_size);
        mBufferBindingBytes.push_back(binding_size * sizeof(float));

        mCpuBuffers[i] = new float[binding_size];

        cudaMalloc(&mGpuBuffers[i], mBufferBindingBytes[i]);

        if (mEngine->bindingIsInput(i))
        {
            mInputDims.push_back(mEngine->getBindingDimensions(i));
        }
        else
        {
            mOutputDims.push_back(mEngine->getBindingDimensions(i));
        }
    }

    CUDA_CHECK(cudaStreamCreate(&mCudaStream));
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool TRTModule::infer()
{
    // Memcpy from host input buffers to device input buffers
    copyInputToDeviceAsync(mCudaStream);

    bool status = mContext->executeV2(mGpuBuffers.data());

    if (!status)
    {
        cout << "inference error!" << endl;
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    copyOutputToHostAsync(mCudaStream);

    return true;
}

//!
//! \brief Copy the contents of input host buffers to input device buffers asynchronously.
//!
void TRTModule::copyInputToDeviceAsync(const cudaStream_t& stream)
{
    memcpyBuffers(true, false, true, stream);
}

//!
//! \brief Copy the contents of output device buffers to output host buffers asynchronously.
//!
void TRTModule::copyOutputToHostAsync(const cudaStream_t& stream)
{
    memcpyBuffers(false, true, true, stream);
}

void TRTModule::memcpyBuffers(const bool copyInput, const bool deviceToHost, const bool async, const cudaStream_t& stream)
{
    for (int i = 0; i < mEngine->getNbBindings(); i++)
    {
        void* dstPtr = deviceToHost ? mCpuBuffers[i] : mGpuBuffers[i];
        const void* srcPtr = deviceToHost ? mGpuBuffers[i] : mCpuBuffers[i];
        const size_t byteSize = mBufferBindingBytes[i];
        const cudaMemcpyKind memcpyType = deviceToHost ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;

        if ((copyInput && mEngine->bindingIsInput(i)) || (!copyInput && !mEngine->bindingIsInput(i)))
        {
            if (async)
            {
                CUDA_CHECK(cudaMemcpyAsync(dstPtr, srcPtr, byteSize, memcpyType, stream));
            }
            else
            {
                CUDA_CHECK(cudaMemcpy(dstPtr, srcPtr, byteSize, memcpyType));
            }
        }
    }
}

size_t TRTModule::getSizeByDim(const Dims& dims)
{
    size_t size = 1;
    
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] == -1)
            size *= MAX_NUM_PROMPTS;
        else
            size *= dims.d[i];
    }

    return size;
}

void TRTModule::setInput(Mat& image)
{
    const int inputH = mInputDims[0].d[2];
    const int inputW = mInputDims[0].d[3];
    
    int i = 0;
    for (int row = 0; row < image.rows; ++row)
    {
        uchar* uc_pixel = image.data + row * image.step;
        for (int col = 0; col < image.cols; ++col)
        {
            mCpuBuffers[0][i] = ((float)uc_pixel[2] / 255.0f - 0.485f) / 0.229f;
            mCpuBuffers[0][i + image.rows * image.cols] = ((float)uc_pixel[1] / 255.0f - 0.456f) / 0.224f;
            mCpuBuffers[0][i + 2 * image.rows * image.cols] = ((float)uc_pixel[0] / 255.0f - 0.406f) / 0.225f;
            uc_pixel += 3;
            ++i;
        }
    }
}

// Set dynamic input
void TRTModule::setInput(float* features, float* imagePointCoords, float* imagePointLabels, float* maskInput, float* hasMaskInput, int numPoints)
{
    delete[]  mCpuBuffers[1];
    delete[]  mCpuBuffers[2];
    mCpuBuffers[1] = new float[numPoints * 2];
    mCpuBuffers[2] = new float[numPoints];

    cudaMalloc(&mGpuBuffers[1], sizeof(float) * numPoints * 2);
    cudaMalloc(&mGpuBuffers[2], sizeof(float) * numPoints);

    mBufferBindingBytes[1] = sizeof(float) * numPoints * 2;
    mBufferBindingBytes[2] = sizeof(float) * numPoints;

    memcpy(mCpuBuffers[0], features, mBufferBindingBytes[0]);
    memcpy(mCpuBuffers[1], imagePointCoords, sizeof(float) * numPoints * 2);
    memcpy(mCpuBuffers[2], imagePointLabels, sizeof(float) * numPoints);
    memcpy(mCpuBuffers[3], maskInput, mBufferBindingBytes[3]);
    memcpy(mCpuBuffers[4], hasMaskInput, mBufferBindingBytes[4]);

    // Setting Dynamic Input Shape in TensorRT
    mContext->setOptimizationProfileAsync(0, mCudaStream);
    mContext->setBindingDimensions(1, Dims3{ 1, numPoints, 2 });
    mContext->setBindingDimensions(2, Dims2{ 1, numPoints });
}

void TRTModule::getOutput(float* features)
{
    memcpy(features, mCpuBuffers[1], mBufferBindingBytes[1]);
}

void TRTModule::getOutput(float* iouPrediction, float* lowResolutionMasks)
{    
    memcpy(lowResolutionMasks, mCpuBuffers[5], mBufferBindingBytes[5]);
    memcpy(iouPrediction, mCpuBuffers[6], mBufferBindingBytes[6]);
}
