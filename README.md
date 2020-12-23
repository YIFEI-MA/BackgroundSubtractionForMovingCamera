# BackgroundSubtractionForMovingCamera
Check out all the source code and dataset:  
https://github.com/YIFEI-MA/BackgroundSubtractionForMovingCamera

### Code Structure
test.py contains the demo of current algorithm.  
/output contains all the resulting image illustrating the foreground feature point that we predicted  
/feature image folder contains all the image with all the feature points on it  
/mask folder contains the mask for the video sequence

#### Structure of test.py

Function testing() is the main function of this file, which read data from all the *.npy file which have the transformation matrix
and labels for training. All the data was calculated by matching frame 1 with next 8 frames.
Then call the function classifier to train the model that classifies the foreground and background, then call the function display
to save the predicted results.

### To Run
To run the test.py, here are some Prerequisites:
* numpy
* matplotlib
* sklearn
* scipy
* cv2
* skimage

Note that to run test.py, you need a version of cv2 that supports SIFT/SURF.

Before running, replace the file path to you owen path in line 
