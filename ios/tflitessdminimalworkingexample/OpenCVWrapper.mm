//
//  OpenCVWrapper.mm
//  tflitessdminimalworkingexample
//
//  Created by Richard Baxter on 26/8/18.
//  Copyright © 2018 Richard Baxter. All rights reserved.
//


#import "OpenCVWrapper.hpp"

#ifdef __cplusplus
#undef NO
#undef YES
#import <opencv2/opencv.hpp>
#import <opencv2/imgcodecs/ios.h>
#endif

#include "object_detect_lite-ios.hpp"

#define SSD_NEURAL_NETWORK_TENSORFLOW_LITE_MODEL_IS_QUANTIZED false
#define SSD_NEURAL_NETWORK_TENSORFLOW_LITE_MODEL_FILE_NAME "coco10.tflite"
#define SSD_NEURAL_NETWORK_TENSORFLOW_LITE_MODEL_FILE_NAME_BASE "coco10"
#define SSD_NEURAL_NETWORK_TENSORFLOW_LITE_MODEL_FILE_NAME_EXTENSION "tflite"
#define SSD_NEURAL_NETWORK_TENSORFLOW_LITE_LABELS_FILE_NAME "coco10_labels.txt"
#define SSD_NEURAL_NETWORK_TENSORFLOW_LITE_LABELS_FILE_NAME_BASE "coco10_labels"
#define SSD_NEURAL_NETWORK_TENSORFLOW_LITE_LABELS_FILE_NAME_EXTENSION "txt"
#define SSD_NEURAL_NETWORK_TENSORFLOW_LITE_MIN_DETECTION_PROBABILITY (0.6)


ObjDetector* objDetector;



@implementation OpenCVWrapper


+ (NSString*)customStringFromFile:(NSString *)fileNameBase andExtension:(NSString *)fileNameExtension
{
	NSString *path = [OpenCVWrapper FilePathForResourceName:fileNameBase andExtension:fileNameExtension];
	//NSString *path = [[NSBundle mainBundle] pathForResource:fileNameBase ofType:fileNameExtension];	//FilePathForResourceName
	NSString* fileContentNS = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:NULL];

	NSLog(@"\n fileContentNS = %@", fileContentNS);
	return fileContentNS;
}

+ (NSString*)FilePathForResourceName:(NSString *)name andExtension:(NSString *)extension;
{
	NSString* file_path = [[NSBundle mainBundle] pathForResource:name ofType:extension];
	if (file_path == NULL) {
		NSLog(@"Couldn't find '%@.%@' in bundle.", name, extension);
		exit(-1);
	}
	return file_path;
}

+ (BOOL)performObjectDetectionInitialiseWrapper
{
	bool result = true;
	
	objDetector = new ObjDetector();
	std::string modelFileName = SSD_NEURAL_NETWORK_TENSORFLOW_LITE_MODEL_FILE_NAME;
	std::string labelsFileName = SSD_NEURAL_NETWORK_TENSORFLOW_LITE_LABELS_FILE_NAME;
	bool modelIsQuantized = SSD_NEURAL_NETWORK_TENSORFLOW_LITE_MODEL_IS_QUANTIZED;
	std::vector<std::string> labels;
	int numberOfLinesInLabelsFile;

	NSString* graph = @SSD_NEURAL_NETWORK_TENSORFLOW_LITE_MODEL_FILE_NAME_BASE;
	const NSString* graph_path = [OpenCVWrapper FilePathForResourceName:graph andExtension:@SSD_NEURAL_NETWORK_TENSORFLOW_LITE_MODEL_FILE_NAME_EXTENSION];
	
	NSString* labelsContentNS = [OpenCVWrapper customStringFromFile:@SSD_NEURAL_NETWORK_TENSORFLOW_LITE_LABELS_FILE_NAME_BASE andExtension:@SSD_NEURAL_NETWORK_TENSORFLOW_LITE_LABELS_FILE_NAME_EXTENSION];
	std::string labelsContent = std::string([labelsContentNS UTF8String]);
	objDetector->getLinesFromFile(&labelsContent, &labels, &numberOfLinesInLabelsFile);
	
	if(!(objDetector->init([graph_path UTF8String], modelIsQuantized, labels)))
	{
		printf("\nfailed to load model");
		result = false;
	}
	
	return result;
}

+ (BOOL)performObjectDetection:(UIImage **)uiImage
{
	BOOL foundObject = false;
	
	cv::Mat imageMatIn;
	cv::Mat imageMatOut;
	UIImageToMat(*uiImage, imageMatIn);
	cvtColor(imageMatIn, imageMatIn, CV_BGRA2BGR);
	
	std::vector<Object> objects;
	objDetector->runImage(imageMatIn, imageMatOut, objects);
	
	*uiImage = MatToUIImage(imageMatOut);
	
	if(objects.size() > 0)
	{
		foundObject = true;
	}
	
	return foundObject;
}





@end
