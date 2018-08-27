//
//  OpenCVWrapper.hpp
//  tflitessdminimalworkingexample
//
//  Created by Richard Baxter on 26/8/18.
//  Copyright © 2018 Richard Baxter. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>



NS_ASSUME_NONNULL_BEGIN


@interface OpenCVWrapper : NSObject

+ (NSString*)customStringFromFile:(NSString *)fileNameBase andExtension:(NSString *)fileNameExtension;
+ (NSString*)FilePathForResourceName:(NSString *)name andExtension:(NSString *)extension;
+ (BOOL)performObjectDetectionInitialiseWrapper;
+ (BOOL)performObjectDetection:(UIImage **)uiImage;

@end

NS_ASSUME_NONNULL_END
