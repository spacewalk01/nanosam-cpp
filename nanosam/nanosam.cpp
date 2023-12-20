#include "nanosam.h"
#include "config.h"

using namespace std;

// Constructor
NanoSam::NanoSam(string encoderPath, string decoderPath)
{
    mImageEncoder = new TRTModule(encoderPath,
        { "image" },
        { "image_embeddings" }, false, true);

    mMaskDecoder = new TRTModule(decoderPath,
        { "image_embeddings", "point_coords", "point_labels", "mask_input", "has_mask_input" },
        { "iou_predictions", "low_res_masks" }, true, false);

    mFeatures = new float[HIDDEN_DIM * FEATURE_WIDTH * FEATURE_HEIGHT];
    mMaskInput = new float[HIDDEN_DIM * HIDDEN_DIM];
    mHasMaskInput = new float;
    mIouPrediction = new float[NUM_LABELS];
    mLowResMasks = new float[NUM_LABELS * HIDDEN_DIM * HIDDEN_DIM];
}

// Deconstructor
NanoSam::~NanoSam()
{
    if (mFeatures)      delete[] mFeatures;
    if (mMaskInput)     delete[] mMaskInput;
    if (mIouPrediction) delete[] mIouPrediction;
    if (mLowResMasks)   delete[] mLowResMasks;

    if (mImageEncoder)  delete mImageEncoder;
    if (mMaskDecoder)   delete mMaskDecoder;
}

// Perform inference using NanoSam models
Mat NanoSam::predict(Mat& image, vector<Point> points, vector<float> labels)
{
    if (points.size() == 0) return cv::Mat(image.rows, image.cols, CV_32FC1);

    // Preprocess encoder input
    auto resizedImage = resizeImage(image, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT);

    // Encoder Inference
    mImageEncoder->setInput(resizedImage);
    mImageEncoder->infer();
    mImageEncoder->getOutput(mFeatures);

    // Preprocess decoder input
    auto pointData = new float[2 * points.size()];
    prepareDecoderInput(points, pointData, points.size(), image.cols, image.rows);

    // Decoder Inference
    mMaskDecoder->setInput(mFeatures, pointData, labels.data(), mMaskInput, mHasMaskInput, points.size());
    mMaskDecoder->infer();
    mMaskDecoder->getOutput(mIouPrediction, mLowResMasks);

    // Postprocessing
    Mat imgMask(HIDDEN_DIM, HIDDEN_DIM, CV_32FC1, mLowResMasks);
    upscaleMask(imgMask, image.cols, image.rows);

    delete[] pointData;

    return imgMask;
}

void NanoSam::prepareDecoderInput(vector<Point>& points, float* pointData, int numPoints, int imageWidth, int imageHeight)
{
    float scale = MODEL_INPUT_WIDTH / max(imageWidth, imageHeight);

    for (int i = 0; i < numPoints; i++)
    {
        pointData[i * 2] = (float)points[i].x * scale;
        pointData[i * 2 + 1] = (float)points[i].y * scale;
    }

    for (int i = 0; i < HIDDEN_DIM * HIDDEN_DIM; i++)
    {
        mMaskInput[i] = 0;
    }
    *mHasMaskInput = 0;
}

Mat NanoSam::resizeImage(Mat& img, int inputWidth, int inputHeight)
{
    int w, h;
    float aspectRatio = (float)img.cols / (float)img.rows;

    if (aspectRatio >= 1)
    {
        w = inputWidth;
        h = int(inputHeight / aspectRatio);
    }
    else
    {
        w = int(inputWidth * aspectRatio);
        h = inputHeight;
    }

    Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, INTER_LINEAR);
    Mat out(inputHeight, inputWidth, CV_8UC3, 0.0);
    re.copyTo(out(Rect(0, 0, re.cols, re.rows)));

    return out;
}

void NanoSam::upscaleMask(Mat& mask, int targetWidth, int targetHeight, int size)
{
    int limX, limY;
    if (targetWidth > targetHeight)
    {
        limX = size;
        limY = size * targetHeight / targetWidth;
    }
    else
    {
        limX = size * targetWidth / targetHeight;
        limY = size;
    }

    cv::resize(mask(Rect(0, 0, limX, limY)), mask, Size(targetWidth, targetHeight));
}
