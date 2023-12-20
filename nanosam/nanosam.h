#pragma once

#include <string>
#include "trt_module.h"

class NanoSam
{

public:

    NanoSam(string encoderPath, string decoderPath);

    ~NanoSam();

    Mat predict(Mat& image, vector<Point> points, vector<float> labels);

private:

    // Variables
    float* mFeatures;
    float* mMaskInput;
    float* mHasMaskInput;
    float* mIouPrediction;
    float* mLowResMasks;

    TRTModule* mImageEncoder;
    TRTModule* mMaskDecoder;

    void upscaleMask(Mat& mask, int targetWidth, int targetHeight, int size = 256);
    Mat resizeImage(Mat& img, int modelWidth, int modelHeight);
    void prepareDecoderInput(vector<Point>& points, float* pointData, int numPoints, int imageWidth, int imageHeight);

};

