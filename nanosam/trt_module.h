#pragma once

#include "NvInfer.h"
#include <opencv2/opencv.hpp>

using namespace nvinfer1;
using namespace std;
using namespace cv;

class TRTModule
{

public:

    TRTModule(string modelPath, vector<string> inputNames, vector<string> outputNames, bool isDynamicShape, bool isFP16);

    bool infer();

    void setInput(Mat& image);

    void setInput(float* features, float* imagePointCoords, float* imagePointLabels, float* maskInput, float* hasHaskInput, int numPoints);

    void getOutput(float* iouPrediction, float* lowResolutionMasks);

    void getOutput(float* features);

    ~TRTModule();

private:

    void build(string onnxPath, vector<string> inputNames, vector<string> outputNames, bool isDynamicShape = false, bool isFP16 = false);

    void deserializeEngine(string engineName, vector<string> inputNames, vector<string> outputNames);

    void initialize(vector<string> inputNames, vector<string> outputNames);

    size_t getSizeByDim(const Dims& dims);

    void memcpyBuffers(const bool copyInput, const bool deviceToHost, const bool async, const cudaStream_t& stream = 0);

    void copyInputToDeviceAsync(const cudaStream_t& stream = 0);

    void copyOutputToHostAsync(const cudaStream_t& stream = 0);


    vector<Dims> mInputDims;            //!< The dimensions of the input to the network.
    vector<Dims> mOutputDims;           //!< The dimensions of the output to the network.
    vector<void*> mGpuBuffers;          //!< The vector of device buffers needed for engine execution
    vector<float*> mCpuBuffers;
    vector<size_t> mBufferBindingBytes;
    vector<size_t> mBufferBindingSizes;
    cudaStream_t mCudaStream;

    IRuntime* mRuntime;                 //!< The TensorRT runtime used to deserialize the engine
    ICudaEngine* mEngine;               //!< The TensorRT engine used to run the network
    IExecutionContext* mContext;        //!< The context for executing inference using an ICudaEngine
};
