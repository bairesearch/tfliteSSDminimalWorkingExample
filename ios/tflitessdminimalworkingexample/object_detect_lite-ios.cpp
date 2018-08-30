/*
This code is based on a combination of;
 obj_detect_lite.cc by YijinLiu (14 August 2018): https://github.com/YijinLiu/tf-cpu/commit/11b8a2c8f4fd882c591c05d9cc7fa4ed538e2661
 with updates by WeiboXu (2 July 2018): https://github.com/tensorflow/tensorflow/files/2153563/obj_detect_lite.cc.zip
 See https://github.com/tensorflow/tensorflow/issues/15633 for development discussion
*/

#include "object_detect_lite-ios.hpp"


#define SSD_NEURAL_NETWORK_SHOW_OBJECT
#define SSD_NEURAL_NETWORK_TENSORFLOW_LITE_MIN_DETECTION_PROBABILITY (0.6)
/*
//optional code:
#define SSD_NEURAL_NETWORK_TENSORFLOW_LITE_LABEL_SELECTED_CLASS_NAME "dog"
#define SSD_NEURAL_NETWORK_TENSORFLOW_LITE_LABEL_SELECTED_CLASS_INDEX 7
*/

float prior_box[4][NUM_RESULT];



template<typename T>
T* TensorData(TfLiteTensor* tensor, int batch_index);

template<>
float* TensorData(TfLiteTensor* tensor, int batch_index)
{
	int nelems = 1;
	for (int i = 1; i < tensor->dims->size; i++) nelems *= tensor->dims->data[i];
	switch (tensor->type)
	{
		case kTfLiteFloat32:
		{
			return tensor->data.f + nelems * batch_index;
		}
		default:
		{
			 printf("\nShould not reach here!");
		}
	}
	return nullptr;
}

template<>
uint8_t* TensorData(TfLiteTensor* tensor, int batch_index)
{
	int nelems = 0;
	for (int i = 1; i < tensor->dims->size; i++) nelems *= tensor->dims->data[i];
	switch (tensor->type)
	{
		case kTfLiteUInt8:
		{
			return tensor->data.uint8 + nelems * batch_index;
		}
		default:
		{
			printf("\nShould not reach here!");
		}
	}
	return nullptr;
}


ObjDetector::ObjDetector()
{
	
}

#define CHAR_NEWLINE '\n'
bool ObjDetector::getLinesFromFile(std::string* fileContentsString, std::vector<std::string>* fileLinesList, int* numberOfLinesInFile)
{
	bool result = true;
	
	int fileNameIndex = 0;
	std::string currentFileName = "";
	
	for(int i=0; i<fileContentsString->length(); i++)
	{
		char currentToken = (*fileContentsString)[i];
		if(currentToken == CHAR_NEWLINE)
		{
			fileLinesList->push_back(currentFileName);
			currentFileName = "";
			fileNameIndex++;
		}
		else
		{
			currentFileName = currentFileName + currentToken;
		}
	}
	
	*numberOfLinesInFile = fileNameIndex;
	
	return result;
}


bool ObjDetector::init(const std::string& model_file, bool is_quantized, const std::vector<std::string>& labels)
{
	input_tensor_ = nullptr;
	output_locations_ = nullptr;
	output_classes_ = nullptr;
	output_scores_ = nullptr;
	num_detections_ = nullptr;
	
	//Load model.
	model_ = tflite::FlatBufferModel::BuildFromFile(model_file.c_str());	//VerifyAndBuildFromFile
	if(!model_)
	{
		printf("\nFailed to load model: %s", model_file.c_str());
		return false;
	}

	//Create interpreter.
	tflite::ops::builtin::BuiltinOpResolver resolver;
	tflite::InterpreterBuilder(*model_, resolver)(&interpreter_);
	if(!interpreter_)
	{
		printf("\nFailed to create interpreter!");
		return false;
	}
	if(interpreter_->AllocateTensors() != kTfLiteOk)
	{
		printf("\nFailed to allocate tensors!");
		return false;
	}
	interpreter_->SetNumThreads(1);

	//Find input / output tensors.
	input_tensor_ = interpreter_->tensor(interpreter_->inputs()[0]);
	
	if(is_quantized)
	{
		if (input_tensor_->type != kTfLiteUInt8)
		{
			printf("\nQuantized graph's input should be kTfLiteUInt8!");
			return false;
		}
	}
	else
	{
		if (input_tensor_->type != kTfLiteFloat32)
		{
			printf("\nQuantized graph's input should be kTfLiteFloat32!");
			return false;
		}
	}
	
	// Find output tensors.
	if (interpreter_->outputs().size() != 4)
	{
		printf("\nGraph needs to have 4 and only 4 outputs!");
		return false;
	}
	output_locations_ = interpreter_->tensor(interpreter_->outputs()[0]);
	output_classes_ = interpreter_->tensor(interpreter_->outputs()[1]);
	output_scores_ = interpreter_->tensor(interpreter_->outputs()[2]);
	num_detections_ = interpreter_->tensor(interpreter_->outputs()[3]);
	
	labels_ = labels;
	return true;
}

bool ObjDetector::runImage(cv::Mat& matInput, cv::Mat& matOutput, std::vector<Object>& objects)
{
	matOutput = matInput;
	if(width() != matInput.cols || height() != matInput.rows)
	{
		cv::Mat resized;
		cv::resize(matInput, resized, cv::Size(width(), height()));
		cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
		FeedInMat(resized, 0);
	}
	else
	{
		cv::Mat for_tf;
		cv::cvtColor(matInput, for_tf, cv::COLOR_BGR2RGB);
		FeedInMat(for_tf, 0);
	}
	if(interpreter_->Invoke() != kTfLiteOk)
	{
		return false;
	}
	AnnotateMat(matOutput, 0, objects);
	return true;
}

float ObjDetector::expit(float x)
{
	return 1.f / (1.f + expf(-x));
}

int ObjDetector::width() const
{
	return input_tensor_->dims->data[2];
}

int ObjDetector::height() const
{
	return input_tensor_->dims->data[1];
}

int ObjDetector::input_channels() const
{
	return input_tensor_->dims->data[3];
}


void ObjDetector::FeedInMat(const cv::Mat& mat, int batch_index)
{
	switch (input_tensor_->type)
	{
		case kTfLiteFloat32:
		{
			float* dst = TensorData<float>(input_tensor_, batch_index);
			const int row_elems = width() * input_channels();
			for (int row = 0; row < height(); row++)
			{
				const uchar* row_ptr = mat.ptr(row);
				for (int i = 0; i < row_elems; i++)
				{
					dst[i] = (row_ptr[i] - IMAGE_MEAN) / IMAGE_STD;	//dst[i] = row_ptr[i] / 128.f - 1.f;
				}
				dst += row_elems;
			}
		}
		break;
		case kTfLiteUInt8:
		{
			uint8_t* dst = TensorData<uint8_t>(input_tensor_, batch_index);
			const int row_elems = width() * input_channels();
			for (int row = 0; row < height(); row++)
			{
				memcpy(dst, mat.ptr(row), row_elems);
				dst += row_elems;
			}
		}
		break;
		default:
		{
			printf("\nShould not reach here!");
		}
	}
}

void ObjDetector::AnnotateMat(cv::Mat& mat, int batch_index, std::vector<Object>& objects)
{
	const float* output_locations = TensorData<float>(output_locations_, batch_index);
	const float* output_classes = TensorData<float>(output_classes_, batch_index);
	const float* output_scores = TensorData<float>(output_scores_, batch_index);
	const int num_detections = output_classes_->dims->data[1];
	const int num_classes = output_classes_->dims->data[2];

	for (int d = 0; d < num_detections; d++)
	{
		const int clsId = output_classes[d];
		const std::string cls = labels_[clsId];
		float score = output_scores[d];
		
		const int ymin = output_locations[4 * d] * mat.rows;
		const int xmin = output_locations[4 * d + 1] * mat.cols;
		const int ymax = output_locations[4 * d + 2] * mat.rows;
		const int xmax = output_locations[4 * d + 3] * mat.cols;
		
		int x = xmin;
		int y = ymin;
		int width = (xmax - xmin);
		int height = (ymax - ymin);

		/*
		//optional code:
		if(cls == SSD_NEURAL_NETWORK_TENSORFLOW_LITE_LABEL_SELECTED_CLASS_NAME)
		{
		*/
			int c = clsId;
			{
				// This score cutoff is taken from Tensorflow's demo app.
				// There are quite a lot of nodes to be run to convert it to the useful possibility
				// scores. As a result of that, this cutoff will cause it to lose good detections in
				// some scenarios and generate too much noise in other scenario.
				if(score >= SSD_NEURAL_NETWORK_TENSORFLOW_LITE_MIN_DETECTION_PROBABILITY)
				{
					Object object;
					object.class_id = c;
					object.rec.x = x;
					object.rec.y = y;
					object.rec.width = width;
					object.rec.height = height;
					object.prob = score;
					objects.push_back(object);
				}
			}
		/*
		}
		*/
	}

	nms(objects);
	
	#ifdef SSD_NEURAL_NETWORK_SHOW_OBJECT
	for(int i = 0; i < objects.size(); i++)
	{
		Object object = objects.at(i);
		//std::cout << "object i = " << i << ", object.prob = " << object.prob << std::endl;
		//std::cout << "object i = " << i << ", object.class_id = " << object.class_id << std::endl;
		
		cv::rectangle(mat, object.rec, cv::Scalar(0, 0, 255), 1);
		cv::putText(mat, labels_[object.class_id], cv::Point(object.rec.x, object.rec.y - 5), cv::FONT_HERSHEY_COMPLEX, .8, cv::Scalar(10, 255, 30));
	}
	#endif
}


static float iou(cv::Rect& rectA, cv::Rect& rectB)
{
	int x1 = std::max(rectA.x, rectB.x);
	int y1 = std::max(rectA.y, rectB.y);
	int x2 = std::min(rectA.x + rectA.width, rectB.x + rectB.width);
	int y2 = std::min(rectA.y + rectA.height, rectB.y + rectB.height);
	int w = std::max(0, (x2 - x1 + 1));
	int h = std::max(0, (y2 - y1 + 1));
	float inter = w * h;
	float areaA = rectA.width * rectA.height;
	float areaB = rectB.width * rectB.height;
	float o = inter / (areaA + areaB - inter);
	return (o >= 0) ? o : 0;
}

static bool objectComp(Object a, Object b)
{
	return a.prob > b.prob;
}

static void nms(std::vector<Object>& boxes)
{
	sort(boxes.begin(), boxes.end(), objectComp);

	std::vector<bool> del(boxes.size(), false);
	for(size_t i = 0; i < boxes.size(); i++)
	{
		if(!del[i])
		{
			for(size_t j = i+1; j < boxes.size(); j++)
			{
				if(iou(boxes.at(i).rec, boxes.at(j).rec) > threshold_iou)
				{
					//LOG(INFO) << "need to delete from the list " << j;
					del[j] = true;
				}
			}
		}
	}

	std::vector<Object> new_boxes;

	for(size_t i = 0; i < boxes.size(); i++)
	{
		if(!del[i])
		{
			new_boxes.push_back(boxes[i]);
		}
	}

	boxes = new_boxes;
}
