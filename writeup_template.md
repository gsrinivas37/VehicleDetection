## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of some of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HSV` and `YCrCb` color space with HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried exploring RBG, HSV and YCrCb channels for extracting HOG features. I found that my accuracy was best when I chose YCrCb with "ALL" channels. 

HSV with only "S" channel gave me ~94% accuracy.
HSV with "ALL" channels gave me ~98% accuracy.
YCrCb with "ALL" channels gave me 99+% accuracy.

Looking at the HOG parameter images, it was visually clear that HSV "S" channel has lot of information. In "YCrCb" channels, all channels seems to have some features. It was not very clearly apparant from images, but after trying different choices and also from hints from lecture videos, I chose "YCrCb" with "ALL" channels.

For other parameters, I chose `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` as recommended in lecture videos.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using HOG features, Spatial features and Histogram features.

The parameters used for HOG feature are: 
* color_space = 'YCrCb'
* orient = 9  
* pix_per_cell = 8
* cell_per_block = 2
* hog_channel = "ALL

I used for spatial size of (32,32) for Spatial feature and bin size of 32 for Histogram features.
The total number of features came out to be 8460. Refer to code cell [8]

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used sliding window search along with Hug sub-sampling technique to avoid computing hog features for each window as described in lecture videos. I used find_cars method in code cell [9] to search multiple scaled windows. I used 75% overlap between windows.

To avoid searching too many windows, I used a different scale in different y value ranges. scale of 1 corresponds to 64x64 pixel windows. I also computed how many windows I'm searching at differnt scale.

* scale of 1.25 between y values 360 and 500  (180 windows)
* scale of 1.5 between 360 and 550 (147 windows)
* scale of 1.75 between 360 and 600 (164 windows)
* scale of 2 between 360 and 656 (72 windows)

I experimented with different scales and y value ranges to optimize number of windows to search and good vehicle detection and found above values optimal.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  
Here's an example result showing positive detections of test_images, created heatmap and the bounding boxes then overlaid on the test_images:

### Here are four test images with positive detections, their corresponding heatmaps and resulting bounding boxes

![alt text][image5]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

* I started with exploring the techniques of Color histogram features, Binned Color features and HOG features individually.
* I explored the above features on car and non-car images and saw how they extracted the relevant details of images.
* I created different utility methods from the lecture videos as I learned these concepts which I eventually used in the project.
* I trained Linear SVM with above features by experimenting with different parameters of HOG and finally arrived at 99% accuracy.
* I used sliding window techinique to search for cars on the test images at different scales. I used multiple sized windows and experimented with scales till I found optimal number of windows and good detections.
* I created heatmap with all the positive detections and used a threshold of 2 to get bounding boxes for indivual cars.
* Finally for the video, I accumulated all the positive detections over 10 consecutive frames to smooth out the boundig boxes and avoid spurious false detections.
* I found that when two cars are closer to each other, instead of detecting two individual cars, it was detected as on big bounding box for two cars.



