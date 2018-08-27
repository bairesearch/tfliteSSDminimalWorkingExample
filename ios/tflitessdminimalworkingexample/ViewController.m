//
//  ViewController.m
//  tflitessdminimalworkingexample
//
//  Created by Richard Baxter on 26/8/18.
//  Copyright © 2018 user. All rights reserved.
//

#import "ViewController.h"

#import "OpenCVWrapper.hpp"

@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad {
	[super viewDidLoad];
	// Do any additional setup after loading the view, typically from a nib.
	
	BOOL result = [OpenCVWrapper performObjectDetectionInitialiseWrapper];
	if(result)
	{
		UIImage* uiImage = [UIImage imageNamed:@"grace_hopper.jpg"];
		
		_imageViewInstance0.image = uiImage;
		BOOL foundObjects = [OpenCVWrapper performObjectDetection:&uiImage];
		
		if(foundObjects)
		{
			_imageViewInstance0.image = uiImage;
		}
	}
}


@end
