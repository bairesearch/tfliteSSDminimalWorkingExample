/*
 This code is based on a combination of;
 obj_detect_lite.cc by YijinLiu (14 August 2018): https://github.com/YijinLiu/tf-cpu/commit/11b8a2c8f4fd882c591c05d9cc7fa4ed538e2661
 with updates by WeiboXu (2 July 2018): https://github.com/tensorflow/tensorflow/files/2153563/obj_detect_lite.cc.zip
 See https://github.com/tensorflow/tensorflow/issues/15633 for development discussion
 */

#ifndef HEADER_OBJECT_DETECT_LITE_IOS
#define HEADER_OBJECT_DETECT_LITE_IOS


#include <math.h>
#include <stdio.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <tensorflow/contrib/lite/kernels/register.h>
#include <tensorflow/contrib/lite/model.h>

#define IMAGE_MEAN 128.0f
#define IMAGE_STD 128.0f

#define NUM_RESULT 1917
#define Y_SCALE    10.0f
#define X_SCALE    10.0f
#define H_SCALE    5.0f
#define W_SCALE    5.0f
#define threshold_iou (0.5f)


template<typename T>
T* TensorData(TfLiteTensor* tensor, int batch_index);
template<>
float* TensorData(TfLiteTensor* tensor, int batch_index);
template<>
uint8_t* TensorData(TfLiteTensor* tensor, int batch_index);


struct Object
{
	cv::Rect rec;
	int      class_id;
	float    prob;
};


class ObjDetector
{
public:
	ObjDetector(void);
	bool getLinesFromFile(std::string* fileContentsString, std::vector<std::string>* fileLinesList, int* numberOfLinesInFile);
	bool init(const std::string& model_file, bool is_quantized, const std::vector<std::string>& labels);
	bool runImage(cv::Mat& matInput, cv::Mat& matOutput, std::vector<Object>& objects);

private:
	
	int width() const;
	int height() const;
	int input_channels() const;

	float expit(float x);
	void FeedInMat(const cv::Mat& mat, int batch_index);
	void AnnotateMat(cv::Mat& mat, int batch_index, std::vector<Object>& objects);

	std::unique_ptr<tflite::FlatBufferModel> model_;
	std::unique_ptr<tflite::Interpreter> interpreter_;
	std::vector<std::string> labels_;

	TfLiteTensor* input_tensor_;
	TfLiteTensor* output_locations_;
	TfLiteTensor* output_classes_;
	TfLiteTensor* output_scores_;
	TfLiteTensor* num_detections_;
};


//static void LoadPriorBoxes();
static float iou(cv::Rect& rectA, cv::Rect& rectB);
static bool objectComp(Object a, Object b);
static void nms(std::vector<Object>& boxes);
//int main(int argc, char** argv);

#endif
