#include "nanosam/nanosam.h"
#include "utils.h"

void segmentClickedPoint(NanoSam& nanosam, string imagePath) {

    auto image = imread(imagePath);

    // Create a window to display the image
    cv::namedWindow("Image");

    // Data structure to hold clicked point
    PointData pointData;
    pointData.clicked = false;
    cv::Mat clonedImage = image.clone();
    int clickCount = 0;

    // Loop until Esc key is pressed
    while (true)
    {
        // Display the original image
        cv::imshow("Image", clonedImage);

        if (pointData.clicked)
        {
            pointData.clicked = false; // Reset clicked flag            

            auto mask = nanosam.predict(image, { pointData.point }, { 1.0f });

            cv::circle(clonedImage, pointData.point, 5, cv::Scalar(0, 0, 255), -1);

            if (clickCount >= CITYSCAPES_COLORS.size()) clickCount = 0;
            overlay(image, mask, CITYSCAPES_COLORS[clickCount * 9]);
            clickCount++;
        }

        // Set the callback function for mouse events on the displayed cloned image
        cv::setMouseCallback("Image", onMouse, &pointData);

        // Check for Esc key press
        char key = cv::waitKey(1);
        if (key == 27) // ASCII code for Esc key
        {
            clonedImage = image.clone();
        }
    }
    cv::destroyAllWindows();
}

void segmentBbox(NanoSam& nanosam, string imagePath, string outputPath, vector<Point> bbox)
{
    auto image = imread(imagePath);

    // 2 : Bounding box top-left, 3 : Bounding box bottom-right
    vector<float> labels = { 2, 3 }; 

    auto mask = nanosam.predict(image, bbox, labels);

    overlay(image, mask);

    rectangle(image, bbox[0], bbox[1], cv::Scalar(255, 255, 0), 3);

    imwrite(outputPath, image);
}

void segmentWithPoint(NanoSam& nanosam, string imagePath, string outputPath, Point promptPoint)
{
    auto image = imread(imagePath);
    
    // 1 : Foreground
    vector<float> labels = { 1.0f }; 

    auto mask = nanosam.predict(image, { promptPoint }, labels);

    overlay(image, mask);

    imwrite(outputPath, image);
}

int main()
{
    /* 1. Load engine examples */

    // Option 1: Load the engines
    //NanoSam nanosam("data/resnet18_image_encoder.engine",  "data/mobile_sam_mask_decoder.engine");

    // Option 2: Build the engines from onnx files
    NanoSam nanosam("data/resnet18_image_encoder.onnx", "data/mobile_sam_mask_decoder.onnx");

    /* 2. Segmentation examples */
    
    // Demo 1: Segment using a point
    segmentWithPoint(nanosam, "assets/dog.jpg", "assets/dog_mask.jpg", Point(1300, 900));
    
    // Demo 2: Segment using a bounding box
    segmentBbox(nanosam, "assets/dogs.jpg", "assets/dogs_mask.jpg", { Point(100, 100), Point(750, 759) });

    // Demo 3: Segment the clicked object
    segmentClickedPoint(nanosam, "assets/dogs.jpg");

    return 0;
}
