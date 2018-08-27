/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.example.user.tflitessdminimalworkingexample;


//activity;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.TextView;
import android.widget.LinearLayout;

//kernel;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.widget.Toast;

//image;
import android.widget.ImageView;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.IOException;

//detector;
import android.app.Activity;
import android.graphics.Color;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.os.SystemClock;
import android.util.Size;
import android.util.TypedValue;
import android.view.Surface;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import java.util.Vector;
import com.example.user.tflitessdminimalworkingexample.OverlayView.DrawCallback;
import com.example.user.tflitessdminimalworkingexample.env.BorderedText;
import com.example.user.tflitessdminimalworkingexample.env.ImageUtils;
import com.example.user.tflitessdminimalworkingexample.env.Logger;
import com.example.user.tflitessdminimalworkingexample.tracking.MultiBoxTracker;


public class MainActivity extends AppCompatActivity
{
	static final boolean AF_NEURAL_NETWORK_SHOW_OBJECT = true;
	static final boolean AF_NEURAL_NETWORK_SHOW_OBJECT_TRACKING = true;	//not yet working (requires libtensorflow_demo.so)

	//frame data (this requires to be set by developer);
	boolean readyToProcessFrame = false;
	Bitmap currentFrameBmp;
	int rotation = 0;
	private static ImageView imgView;

	//kernel;
	static int frameIndex = 1;
	
	//detector;
	//extracted from CameraActivity.java;
	private boolean debug = false;
	protected int previewWidth = 0;
	protected int previewHeight = 0;
	//#ifdef AF_NEURAL_NETWORK_SHOW_OBJECT_TRACKING
	OverlayView trackingOverlay;
	//#endif

	//extracted from DetectorActivity.java;
	private static final Logger LOGGER = new Logger();

	// Configuration values for the prepackaged SSD model.
	private static final int TF_OD_API_INPUT_SIZE = 300;
	private static final boolean TF_OD_API_IS_QUANTIZED = true;
	private static final String TF_OD_API_MODEL_FILE = "detect.tflite";
	private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/coco_labels_list.txt";

	// Which detection model to use: by default uses Tensorflow Object Detection API frozen
	// checkpoints.
	private enum DetectorMode 
	{
		TF_OD_API;
	}

	private static final MainActivity.DetectorMode MODE = MainActivity.DetectorMode.TF_OD_API;

	// Minimum detection confidence to track a detection.
	private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.6f;

	private static final boolean MAINTAIN_ASPECT = false;

	private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);

	private static final boolean SAVE_PREVIEW_BITMAP = false;
	private static final float TEXT_SIZE_DIP = 10;

	private Integer sensorOrientation;

	private Classifier detector;

	private long lastProcessingTimeMs;
	private Bitmap rgbFrameBitmap = null;
	private Bitmap croppedBitmap = null;
	private Bitmap cropCopyBitmap = null;

	private boolean computingDetection = false;

	private long timestamp = 0;

	private Matrix frameToCropTransform;
	private Matrix cropToFrameTransform;

	private MultiBoxTracker tracker;

	private byte[] luminanceCopy;

	private BorderedText borderedText;


	//from CameraActivity.java;
	private Handler handler;
	private HandlerThread handlerThread;

	private final static String TAG = MainActivity.class.getSimpleName();


	public Bitmap getBitmapFromAssetsFolder(String fileName)
	{
		Bitmap bitmap = null;
		try
		{
			InputStream istr=getAssets().open(fileName);
			bitmap=BitmapFactory.decodeStream(istr);
		}
		catch (IOException e1)
		{
			// TODO Auto-generated catch block
			//e1.printStackTrace();
			System.out.println("Error: " + e1);
			System.exit(0);
		}
		return bitmap;
	}

	@Override
	protected void onCreate(Bundle savedInstanceState) 
	{
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);

		Bitmap imageBitMap = getBitmapFromAssetsFolder("grace_hopper.jpg");
		int imageWidth = imageBitMap.getWidth();
		int imageHeight = imageBitMap.getHeight();
		imgView = findViewById(R.id.imageView1);
		imgView.setImageBitmap(imageBitMap);
		currentFrameBmp = imageBitMap;
		readyToProcessFrame = true;

		initialiseDetector(imageWidth, imageHeight, rotation); //neural network object recognition

		executeKernel();

	}

	private void executeKernel()
	{
		/*
		Bitmap currentFrameBmpTemp = currentFrameBmp;

		byte[] currentFrameLuminance = calculateLuminanceMap(currentFrameBmpTemp);
		processImage(currentFrameLuminance, currentFrameBmpTemp);
		Log.i(TAG, "processImage");
		*/

		final Handler handlerKernel = new Handler();
		final Runnable r = new Runnable()
		{
			//@Override
			public void run()
			{
				handlerKernel.postDelayed(this, 66);	//executing object detection every second

				if(readyToProcessFrame && !computingDetection)
				{
					Bitmap currentFrameBmpTemp = currentFrameBmp;

					byte[] currentFrameLuminance = calculateLuminanceMap(currentFrameBmpTemp);
					processImage(currentFrameLuminance, currentFrameBmpTemp);
					Log.i(TAG, "processImage");
				}
			}
		};
		handlerKernel.postDelayed(r, 66);
	}



	//detector;

	//#ifdef AF_NEURAL_NETWORK_SHOW_OBJECT_TRACKING
	//object detection/tracking display code;	//CHECKTHIS
	public void requestRender()
	{
		final OverlayView overlay = (OverlayView) findViewById(R.id.debug_overlay);
		if (overlay != null) {
			overlay.postInvalidate();
		}
	}
	public void addCallback(final OverlayView.DrawCallback callback)
	{
		final OverlayView overlay = (OverlayView) findViewById(R.id.debug_overlay);
		if (overlay != null) {
			overlay.addCallback(callback);
		}
	}
	//#endif

	@Override
	protected void onResume()
	{
		super.onResume();

		//detector;
		handlerThread = new HandlerThread("inference");
		handlerThread.start();
		handler = new Handler(handlerThread.getLooper());
	}

	@Override
	protected void onPause()
	{
		super.onPause();

		//detector;
		if (!isFinishing()) {
			LOGGER.d("Requesting finish");
			finish();
		}
		handlerThread.quitSafely();
		try
		{
			handlerThread.join();
			handlerThread = null;
			handler = null;
		}
		catch (final InterruptedException e)
		{
			LOGGER.e(e, "Exception!");
		}
	}


	protected int getScreenOrientation()
	{
		switch (getWindowManager().getDefaultDisplay().getRotation())
		{
			case Surface.ROTATION_270:
				return 270;
			case Surface.ROTATION_180:
				return 180;
			case Surface.ROTATION_90:
				return 90;
			default:
				return 0;
		}
	}
	public boolean isDebug()
	{
		return debug;
	}
	protected int getLuminanceStride()
	{
		return previewWidth;  //yRowStride
	}
	protected synchronized void runInBackground(final Runnable r)
	{
		if (handler != null) {
			handler.post(r);
		}
	}

	//extracted from DetectorActivity.java;
	public void initialiseDetector(final int imageWidth, final int imageHeight, final int rotation)
	{
		final float textSizePx = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
		borderedText = new BorderedText(textSizePx);
		borderedText.setTypeface(Typeface.MONOSPACE);

		tracker = new MultiBoxTracker(this);

		int cropSize = TF_OD_API_INPUT_SIZE;

		try
		{
			detector = TFLiteObjectDetectionAPIModel.create(getAssets(), TF_OD_API_MODEL_FILE, TF_OD_API_LABELS_FILE, TF_OD_API_INPUT_SIZE, TF_OD_API_IS_QUANTIZED);
			cropSize = TF_OD_API_INPUT_SIZE;
		}
		catch (final IOException e)
		{
			LOGGER.e("Exception initializing classifier!", e);
			Toast toast = Toast.makeText(getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
			toast.show();
			finish();
		}

		previewWidth = imageWidth;
		previewHeight = imageHeight;

		sensorOrientation = rotation - getScreenOrientation();
		LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

		LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
		rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
		croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888);

		frameToCropTransform = ImageUtils.getTransformationMatrix(previewWidth, previewHeight, cropSize, cropSize, sensorOrientation, MAINTAIN_ASPECT);

		cropToFrameTransform = new Matrix();
		frameToCropTransform.invert(cropToFrameTransform);

		if(AF_NEURAL_NETWORK_SHOW_OBJECT_TRACKING)
		{
			//object detection/tracking display code;
			trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
			trackingOverlay.addCallback(
					new DrawCallback() {
						@Override
						public void drawCallback(final Canvas canvas) {
							LOGGER.i("drawCallback1");
							tracker.draw(canvas);
							if (isDebug()) {
								tracker.drawDebug(canvas);
							}
						}
					});

			addCallback(
					new DrawCallback() {
						@Override
						public void drawCallback(final Canvas canvas) {
							LOGGER.i("drawCallback2");
							if (!isDebug()) {
								return;
							}
							final Bitmap copy = cropCopyBitmap;
							if (copy == null) {
								return;
							}

							final int backgroundColor = Color.argb(100, 0, 0, 0);
							canvas.drawColor(backgroundColor);

							final Matrix matrix = new Matrix();
							final float scaleFactor = 2;
							matrix.postScale(scaleFactor, scaleFactor);
							matrix.postTranslate(
									canvas.getWidth() - copy.getWidth() * scaleFactor,
									canvas.getHeight() - copy.getHeight() * scaleFactor);
							canvas.drawBitmap(copy, matrix, new Paint());

							final Vector<String> lines = new Vector<String>();
							if (detector != null) {
								final String statString = detector.getStatString();
								final String[] statLines = statString.split("\n");
								for (final String line : statLines) {
									lines.add(line);
								}
							}
							lines.add("");

							lines.add("Frame: " + previewWidth + "x" + previewHeight);
							lines.add("Crop: " + copy.getWidth() + "x" + copy.getHeight());
							lines.add("View: " + canvas.getWidth() + "x" + canvas.getHeight());
							lines.add("Rotation: " + sensorOrientation);
							lines.add("Inference time: " + lastProcessingTimeMs + "ms");

							borderedText.drawLines(canvas, 10, canvas.getHeight() - 10, lines);
						}
					});
		}
	}

	protected void processImage(byte[] originalLuminanceTemp, Bitmap currentFrameBmpTemp)
	{
		++timestamp;
		final long currTimestamp = timestamp;

		tracker.onFrame(
				previewWidth,
				previewHeight,
				getLuminanceStride(),
				sensorOrientation,
				originalLuminanceTemp,
				timestamp);
		if(AF_NEURAL_NETWORK_SHOW_OBJECT_TRACKING)
		{
			trackingOverlay.postInvalidate();
		}

		LOGGER.i("processImage2");
		if(computingDetection)
		{
			LOGGER.i("computingDetection - skip frame");
			//skip frame
			return;
		}
		computingDetection = true;
		LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

		rgbFrameBitmap = currentFrameBmpTemp;

		if (luminanceCopy == null)
		{
			luminanceCopy = new byte[originalLuminanceTemp.length];
		}
		System.arraycopy(originalLuminanceTemp, 0, luminanceCopy, 0, originalLuminanceTemp.length);

		final Canvas canvas = new Canvas(croppedBitmap);
		canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
		// For examining the actual TF input.
		if (SAVE_PREVIEW_BITMAP)
		{
			ImageUtils.saveBitmap(croppedBitmap);
		}

		runInBackground(
				new Runnable()
				{
					@Override
					public void run()
					{
						LOGGER.i("Running detection on image " + currTimestamp);
						final long startTime = SystemClock.uptimeMillis();
						final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
						lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

						cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
						final Canvas canvas = new Canvas(cropCopyBitmap);
						final Paint paint = new Paint();
						paint.setColor(Color.RED);
						paint.setStyle(Paint.Style.STROKE);
						paint.setStrokeWidth(2.0f);

						float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
						switch (MODE)
						{
							case TF_OD_API:
								minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
								break;
						}

						final List<Classifier.Recognition> mappedRecognitions = new LinkedList<Classifier.Recognition>();


						for (final Classifier.Recognition result : results)
						{
							final RectF location = result.getLocation();
							LOGGER.i("object found");
							if (location != null && result.getConfidence() >= minimumConfidence)
							{
								LOGGER.i("passedConfidence");
								canvas.drawRect(location, paint);

								cropToFrameTransform.mapRect(location);
								result.setLocation(location);
								mappedRecognitions.add(result);

							}
						}

						frameIndex++;

						if(AF_NEURAL_NETWORK_SHOW_OBJECT)
						{
							tracker.trackResults(mappedRecognitions, luminanceCopy, currTimestamp);	//this also adds a box around found objects
						}
						if(AF_NEURAL_NETWORK_SHOW_OBJECT_TRACKING)
						{
							//object detection/tracking display code;
							trackingOverlay.postInvalidate();
							requestRender();
						}

						computingDetection = false;
					}
				});
	}


	private byte[] calculateLuminanceMap(Bitmap bitmap)
	{
		int R = 0; int G = 0; int B = 0;
		int height = bitmap.getHeight();
		int width = bitmap.getWidth();
		byte[] luminanceMap = new byte[width * height];
		int[] pixels = new int[width * height];
		bitmap.getPixels(pixels, 0, width, 0, 0, width, height);
		for (int i = 0; i < pixels.length; i++)
		{
			int color = pixels[i];
			R += Color.red(color);
			G += Color.green(color);
			B += Color.blue(color);
			byte luminance = (byte)((R+G+B)/3);
			luminanceMap[i] = luminance;
		}
		return luminanceMap;
	}
}
