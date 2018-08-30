# tfliteSSDminimalWorkingExample
TensorFlow Lite SSD (Object Detection) Minimal Working Example for iOS and Android

Here are instructions for building and running a minimal working example of TensorFlow Lite SSD/object detection on iOS and Android.

How to build and run a minimal working example of TensorFlow Lite SSD/object detection on iOS
================================================================================================

Part 1 (create a new iOS project with OpenCV libraries installed)
-----------------

Note if using the sample project github.com/baxterai/tflitessdminimalworkingexample/ios;
 - delete the existing references (red) to `opencv2.framework` in;
   - Project Navigator - Frameworks
   - Project Navigator - [APPLICATIONNAME] project (blue document icon) - [default target] - Build Phases - Link Binary With Libraries
 - Download the opencv framework (https://opencv.org/releases.html , e.g. https://sourceforge.net/projects/opencvlibrary/files/opencv-ios/3.4.2/opencv-3.4.2-ios-framework.zip/download )
 - Project Navigator - select [APPLICATIONNAME] project (blue document icon) - [default target]
 - Build Phases - Link Binary With Libraries - Click the '+' - Add Other... - browse and select `opencv2.framework` (e.g. `/Users/[USERNAME]/Documents/libraries/opencv2.framework`) 
 - Build Settings - Framework Search Paths - add `[INSERT COMPLETE PATH OF opencv2.framework]` (e.g. `/Users/[USERNAME]/Documents/libraries/`) 
 - Project Navigator - select [APPLICATIONNAME] project (blue document icon) - [default target] - Build Phases
 
Otherwise, create an iOS Objective-C project with OpenCV (these remaining instructions can be skipped if using the sample project github.com/baxterai/tflitessdminimalworkingexample/ios):

[Extract from https://stackoverflow.com/questions/52030819/how-to-create-an-ios-objective-c-project-with-opencv/52030820#52030820 ] (instructions based on https://medium.com/@yiweini/opencv-with-swift-step-by-step-c3cc1d1ee5f1 )
 
 - Download the opencv framework (https://opencv.org/releases.html , e.g. https://sourceforge.net/projects/opencvlibrary/files/opencv-ios/3.4.2/opencv-3.4.2-ios-framework.zip/download )
 - Project Navigator - select [APPLICATIONNAME] project (blue document icon) - [default target]
 - Build Phases - Link Binary With Libraries - Click the '+' - Add Other... - browse and select `opencv2.framework` (e.g. `/Users/[USERNAME]/Documents/libraries/opencv2.framework`) 
 - Build Phases - Link Binary With Libraries - Click the '+' - add these additional frameworks (for opencv); AssetsLibrary, CoreGraphics, CoreMedia, CoreFoundation, Accelerate, [UIKit, Foundation, CoreVideo, CoreImage]
 - Build Settings - Framework Search Paths - add `$(PROJECT_DIR)`
 - Build Settings - Framework Search Paths - add `[INSERT COMPLETE PATH OF opencv2.framework]` (e.g. `/Users/[USERNAME]/Documents/libraries/`) 
 - File - New - File - Cocoa Touch Class. Name it `OpenCVWrapper` and choose objective-C for Language.
 - `OpenCVWrapper.m` and rename the file extension to `.mm`
 - Manually change `OpenCVWrapper.m` to `OpenCVWrapper.mm` in the file header also
 - Go to `OpenCVWrapper.mm` and add the following import statement on the top; `#import <opencv2/opencv.hpp>`

Part 2 (install the TFlite library)
-----------------

Note if using the sample project github.com/baxterai/tflitessdminimalworkingexample/ios;
 - delete the existing references (red) to `libtensorflow-lite.a` in;
   - Project Navigator - Frameworks
   - Project Navigator - [APPLICATIONNAME] project (blue document icon) - [default target] - Build Phases - Link Binary With Libraries
 - build the libtensorflow-lite library and recreate the references to it (implement the first section of the following instructions); 

[Extract from https://stackoverflow.com/questions/52030130/how-to-build-and-run-the-tensorflow-lite-ios-examples/52030131#52030131 ] (instructions based on https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/g3doc/ios.md )

 - `git clone https://github.com/tensorflow/tensorflow`
 - `cd tensorflow`
 - `git checkout r1.10` (select a version of tensorflow that contains tensorflow/contrib/lite/download_dependencies.sh)
   - [these instructions are based on https://github.com/tensorflow/tensorflow/tree/r1.10 ]
 - `cd tensorflow/contrib/lite`
 - `./download_dependencies.sh`
 - `./build_ios_universal_lib.sh`
 - `cd examples/ios`
 - `./download_models.sh` (download tensorflow models to `simple/data` and `camera/data`)
 - Show the project navigator
 - Update various settings in your app to link against TensorFlow Lite (see `tensorflow/contrib/lite/examples/ios/simple/simple.xcodeproj` / `camera/tflite_camera_example.xcworkspace` for example):
   - select [APPLICATIONNAME] project (blue document icon) - [default target] - Build Phases
   - Link Binary With Libraries - add (+) library `tensorflow/contrib/lite/gen/lib/libtensorflow-lite.a`
   - select [APPLICATIONNAME] project (blue document icon) - [default target] - Build Settings
   - Library Search Paths - add `[INSERTTENSORFLOWSOURCELOCATIONHERE]/tensorflow/contrib/lite/gen/lib`
   - Header Search paths - add `[INSERTTENSORFLOWSOURCELOCATIONHERE]` (the root folder of the tensorflow git repository)
   - Header Search paths - add `[INSERTTENSORFLOWSOURCELOCATIONHERE]/tensorflow/contrib/lite/downloads`
   - Header Search paths - add `[INSERTTENSORFLOWSOURCELOCATIONHERE]/tensorflow/contrib/lite/downloads/flatbuffers/include`
   - C++11 support (or later) should be enabled by setting C++ Language Dialect to GNU++11 (or GNU++14), and C++ Standard Library to libc++.

These remaining instructions can be skipped if using the sample project github.com/baxterai/tflitessdminimalworkingexample/ios;

 - project navigator - in [INSERTAPPLICATIONNAME] project (blue document icon) - create new group called `data`
 - drag and drop `data` folder items from `tensorflow/contrib/lite/example/ios/simple/data` (`grace_hopper.jpg`, `labels.txt`, `mobilenet...`) to the newly created data folder in xCode (when asked, select Destination: Copy items if needed)
 - modify the application's existing `AppDelegate.m` accordingly with `tensorflow/contrib/lite/examples/ios/simple/AppDelegate.mm`/`.h` contents
 - drag and drop source code items from `tensorflow/contrib/lite/example/ios/simple` (`ios_image_load.h`, `ios_image_load.mm`, `RunModelViewController.h`, `RunModelViewController.mm`, `RunModelViewController.xib` [NOT: `AppDelegate.h`, `AppDelegate.mm`, `main.mm`])


Part 3 (Create a patched version of `obj_detect_lite.cc`)
-----------------

Note these instructions can be be skipped by using the `obj_detect_lite-ios.cpp` code in github.com/baxterai/tflitessdminimalworkingexample/ios;

Instructions based on; https://github.com/tensorflow/tensorflow/issues/15633 - hidden items - load more

 - manually create a patched version of `obj_detect_lite.cc` 
  - git clone https://github.com/YijinLiu/tf-cpu
  - OPTIONAL: `git checkout master` / `11b8a2c8f4fd882c591c05d9cc7fa4ed538e2661` (these instructions are based on the 14 August 2018 version, https://github.com/YijinLiu/tf-cpu/commit/11b8a2c8f4fd882c591c05d9cc7fa4ed538e2661)
  - method 1;
    - add the reference `nms(objects)` along with the functions `iou`/`objectComp`/`nms`
    - add the `struct Object` defintition along with its references in `class ObjDetector` (`std::vector<Object> objects;`) and function `AnnotateMat` 
  - method 2;
    - copy the latest version of `obj_detect_lite.cc` by YijinLiu `tf-cpu/benchmark/obj_detect_lite.cc` to a temporary folder `vMaster`
    - `git checkout f31fb25d97adcd00124646c0413bfdec989840fa` (checkout the older 28 May 2018 version of https://github.com/YijinLiu/tf-cpu upon which WeiboXu's version was created, such that a diff can be generated)
    - copy the older 28 May 2018 version of `tf-cpu/benchmark/obj_detect_lite.cc` by YijinLiu to a temporary folder `vOrig`
    - copy the patched version of `obj_detect_lite.cc` by WeiboXu from `https://github.com/tensorflow/tensorflow/files/2153563/obj_detect_lite.cc.zip` to a temporary folder `vOrigPatched`
    - `diff vOrig vOrigPatched` (use windiff/kdiff3 for GUI)
    - note the changes to the file which must be integrated into the `vMaster` version of `obj_detect_lite.cc`
  - Alternatively, download the TF Lite SSD minimal working example IOS source from `github.com/baxterai/tflitessdminimalworkingexample/ios`

Part 4 (Prepare the user interface)
-----------------

Note these instructions can be be skipped using the UI code in github.com/baxterai/tflitessdminimalworkingexample/ios;

 - Project Navigator (left pane) - [INSERTAPPLICATIONNAME]
 - Open Main.storyboard
 - Select View Controller Scene - View Controller - View
 - Select the Library button (top right: circle with inside square)
 - Search for 'Image View'
 - Drag and drop a new View Image Object into View Controller Scene - View Controller - View (adjacent 'Safe Area')
 - Create an outlet connection for the new Image View object
 - Select the Assistant Editor button (top right: two intersecting circles)
 - Select file ViewController.h in the Project Navigator
 - If necessary hide the left and right panes to create more space (top right: blue box with left vertical bar, blue box with right vertical bar)
 - hold the Ctrl key and drag the new image view object to the ViewController.h file (immediately below @interface ViewController: UIViewController) - name the object reference as imageViewInstance0

Part 6 (Add the TFLite SSD execution files)
-----------------
 
Note these instructions can be be skipped using the project github.com/baxterai/tflitessdminimalworkingexample/ios;

 - Download the TF Lite SSD minimal working example IOS source from `github.com/baxterai/tflitessdminimalworkingexample/ios`
 - Project Navigator (left pane) - [APPLICATIONNAME]
 - Drag and drop the relevant source code items from `tflitessdminimalworkingexample` (`object_detect_lite-ios.cpp`/`object_detect_lite-ios.hpp`/`OpenCVWrapper.hpp`/`OpenCVWrapper.mm`) into the primary application folder in xCode (when asked, select Destination: Copy items if needed)
 - Merge the relevant code from `tflitessdminimalworkingexample` `ViewController.m` into [APPLICATIONNAME] `ViewController.m`
 - Project navigator - in [APPLICATIONNAME] project (blue document icon) - create new group called `data`
 - Drag and drop `data` folder items from tflitessdminimalworkingexample/data (`grace_hopper.jpg`, `coco10_labels.txt`, `coco10.tflite`) to the newly created data folder in xCode (when asked, select Destination: Copy items if needed)


How to build and run a minimal working example of TensorFlow Lite SSD/object detection on Android
================================================================================================

Note these instructions can be be skipped using the project github.com/baxterai/tflitessdminimalworkingexample/android;

Instructions based on; https://stackoverflow.com/a/51969021/2585501 (https://stackoverflow.com/questions/50330184/build-and-run-tensorflow-lite-demo-with-gradle)

Note these instructions should theoretically support optional object tracking by adding the `libtensorflow_demo.so` library (although adding this library causes the app to crash; it may relate to a conflicting package naming scheme or gradle build conflict)

Part 1 (create a new Android project)
-----------------

Android Studio - create a new Android project
 - Basic 
  - Application name: e.g. [TF Lite SSD Minimal Working Example]
  - Company domanin: e.g. [user.example.com]
  - Project location: e.g. [.../tfliteSSDminimalWorkingExampleAndroid/]
  - Include C++ Support
  - generated package name: [com.example.user.tflitessdminimalworkingexample]
  - Next
 - Select the form factors and minimum SDK
  - Phone and Tablet - e.g. [API 25: Android 7.1.1 (Nougat)] 
  - Next
 - Add an Activity to Mobile
  - Empty Activity
  - Next
- Configure Activity
  - Activity Name: [MainActivity]
  - tick generate layout file
  - Layout Name: `activity_name`
  - Backwards compatibility (AppCompat)
  - Next
- Customise C++ Support
  - C++ Standard: Toolchain Default

Part 2 (prepare the Android project)
-----------------

 - Open the android project (tflitessdminimalworkingexample)
 - edit app/build.gradle
  - add the following code;
  
  android {
  
  ...
  
        lintOptions {
            abortOnError false
        }    
        aaptOptions {
            noCompress "tflite"
        }
    
        compileOptions {
            sourceCompatibility JavaVersion.VERSION_1_8
            targetCompatibility JavaVersion.VERSION_1_8
        }
    }
    
    repositories {
        maven {
            url 'https://google.bintray.com/tensorflow'
        }
    }
    
    // import DownloadModels task
    project.ext.ASSET_DIR = projectDir.toString() + '/src/main/assets'
    project.ext.TMP_DIR   = project.buildDir.toString() + '/downloads'
    
    // Download default models; if you wish to use your own models then
    // place them in the "assets" directory and comment out this line.
    apply from: "download-models.gradle"

    dependencies {
    
    ...

        implementation 'org.tensorflow:tensorflow-lite:+'

Part 3 (copy the minimal TensorFlow Lite Android example source code)
-----------------	

 - `git clone https://github.com/tensorflow/tensorflow`
 - `cd tensorflow`
 - OPTIONAL: `git checkout master` / `938a3b77797164db736a1006a7656326240baa59` 
   - [these instructions are based on https://github.com/tensorflow/tensorflow/commit/938a3b77797164db736a1006a7656326240baa59 ]
 - open `tensorflow/contrib/lite/examples/android/app/src/main/java/org/tensorflow/demo/` in file explorer
 - copy `AutoFitTextureView.java`, `Classifier.java`, `OverlayView.java`, `TFLiteObjectDetectionAPIModel.java` to `[PATHTOAPPLICATION]/app/src/main/java/com/example/user/[APPLICATIONNAME]`
 - copy `AutoFitTextureView.java`, `Classifier.java`, `OverlayView.java`, `TFLiteObjectDetectionAPIModel.java` to `[PATHTOAPPLICATION]/app/src/main/java/com/example/user/[APPLICATIONNAME]`
 - copy env `BorderedText.java`, `ImageUtils.java`, `Logger.java`, `Size.java` to `[PATHTOAPPLICATION]/app/src/main/java/com/example/user/[APPLICATIONNAME]/env`
 - copy env `MultiBoxTracker.java`, `ObjectTracker.java` to `[PATHTOAPPLICATION]/app/src/main/java/com/example/user/tflitessdminimalworkingexample/tracking`
 - in all copied java files, replace `package org.tensorflow.demo.env` (e.g. with `package com.example.user.[APPLICATIONNAME]` where this matches the `[PATHTOAPPLICATION]/app/src/main/java/com/example/user/[APPLICATIONNAME]` directory structure)
 - in all copied java files, replace all instances of `com.example.user.tflitessdminimalworkingexample` (e.g. with `com.example.user.[APPLICATIONNAME]` where this matches the `[PATHTOAPPLICATION]/app/src/main/java/com/example/user/[APPLICATIONNAME]` directory structure)

Part 4 (copy the minimal TensorFlow Lite Android example UI code)
-----------------
	
 - modify `[PATHTOAPPLICATION]/app/src/main/res/activity_main.xml`
  - change `android.support.constraint.ConstraintLayout` to `android.support.design.widget.CoordinatorLayout` in `[PATHTOAPPLICATION]/app/src/main/res/activity_main.xml`;
  - in 'Design' (or 'Text') view, add an ImageView object called imageView1 to the component tree - CoordinatorLayout
  - open `tensorflow/contrib/lite/examples/android/app/src/main/res/` in file explorer
  - import the `layoutcamera_connection_fragment_tracking.xml` contents into `[PATHTOAPPLICATION]/app/src/main/res/activity_main.xml`;
 
        <?xml version="1.0" encoding="utf-8"?>
        <android.support.design.widget.CoordinatorLayout xmlns:android="http://schemas.android.com/apk/res/android"
            xmlns:app="http://schemas.android.com/apk/res-auto"
            xmlns:tools="http://schemas.android.com/tools"
            android:layout_width="match_parent"
            android:layout_height="match_parent"
            tools:context=".MainActivity">
        
            <ImageView
                android:id="@+id/imageView1"
                android:scaleType="fitStart"
                android:layout_width="match_parent"
                android:layout_height="wrap_content"
                app:srcCompat="@android:color/holo_green_dark" />
        
            <com.example.user.[APPLICATIONNAME].OverlayView
                android:id="@+id/tracking_overlay"
                android:layout_width="match_parent"
                android:layout_height="match_parent"/>
        
            <com.example.user.[APPLICATIONNAME].OverlayView
                android:id="@+id/debug_overlay"
                android:layout_width="match_parent"
                android:layout_height="match_parent"/>
    
        </android.support.design.widget.CoordinatorLayout>
    
Part 5 (create the TensorFlow Lite Android example UI minimal activity source code)
-----------------

 - Download the TF Lite SSD minimal working example IOS source from github.com/baxterai/tflitessdminimalworkingexample/android
 - insert the `tfliteSSDminimalWorkingExample/app/src/main/java/com/example/user/tflitessdminimalworkingexample/MainActivity.java` contents into `[PATHTOAPPLICATION]/app/src/main/java/com/example/user/tflitessdminimalworkingexample/MainActivity.java
 - Gradle Sync
 - Build
 - Run
 - [give permissions to app when requested]
 - Run example app (tflitessdminimalworkingexample) on Android phone (search - tflitessdminimalworkingexample)
