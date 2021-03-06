# **Traffic Sign Recognition** 

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./nopassing.jpg "Traffic Sign 1"
[image5]: ./bumpyroad.jpg "Traffic Sign 2"
[image6]: ./speed30.jpg "Traffic Sign 3"
[image7]: ./roundaboutmandatory.jpg "Traffic Sign 4"
[image8]: ./roadwork.jpg "Traffic Sign 5"
[image9]: ./resized.png "Resized traffic sizes"

## Rubric
### The rubric for this project is [here](https://review.udacity.com/#!/rubrics/481/view).

---
### Files submitted

1. [`Traffic_Sign_Classifier.ipynb`](Traffic_Sign_Classifier.ipynb)
2. [`Traffic_Sign_Classifier.html`](Traffic_Sign_Classifier.html)
3. [`writeup.md`](writeup.md)

### Data Set Summary & Exploration

#### 1. Summary

The dataset was explored using `numpy`.

* The size of training set is **34799**
* The size of the validation set is **4410**
* The size of test set is **12630**
* The shape of a traffic sign image is **(32, 32, 3)**
* The number of unique classes/labels in the data set is **43**

#### 2. Label distribution.

Here are bar charts that plot the distribution of examples vs labels in the train, validate and test datasets.

![alt text][image1]

### Design and Test a Model Architecture

#### Preprocessing
1. Images are converted to grayscale using `rgb2grayscale` function. 
Images are grayscaled as the color in the signs does not matter as much as the shapes in the signs. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

2. The pixels in the image were normalized to have a mean of zero and equal variance using this formula. NNs perform better on datasets which have zero mean and equal variance.

> pixel = (pixel - 128) / 128


#### 2. Model

The model consists of the following transforms and layers:

| Layer         		|      Description	        					| Output shape
|:---------------------:|:---------------------------------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							| `32 x 32 x 3`
| `rgb2grayscale` | Convert the image to grayscale. Reduce 3 channels to 1. | `32 x 32 x 1`|
| `normalize_pixel_v` | Normalize the pixel values to have a zero mean and equal variance | `32 x 32 x 1` |
| Convolution `5 x 5 x 1`    	| `1 x 1` stride, `valid` padding	| `28 x 28 x 6` |
| RELU					|				Activation 								| `28 x 28 x 6` |
| Max pooling	 `2 x 2 x 1`      	|  `2 x 2` stride,  `valid` padding | `14 x 14 x 6` 				|
| Convolution `5 x 5	x 6`    |  `1 x 1` stride, `valid` padding  									| `10 x 10 x 16` |
| RELU					|				Activation 								| `10 x 10 x 16` |
| Max pooling	 `2 x 2 x 1`      	|  `2 x 2` stride,  `valid` padding | `5 x 5 x 16` 				|
| Flatten | &nbsp; | `1 x 400` |
| Fully connected		| 120 neurons in hidden layer      									|  `1 x 120` |
| RELU					|				Activation 								| `1 x 120` |
| Fully connected		| 84 neurons in hidden layer     									|  `1 x 84` |
| RELU					|				Activation 								| `1 x 84` |
| Fully connected		| 43 neurons in output layer. One for each of the traffic signs.    									|  `1 x 43` |
 


#### 3. Hyperparamters and Optimizer

| Hyperparamters | value |
|:--------------:|:-----:|
| Batch Size     | 128 |
| Epochs         | 45  |
| Learning rate  | 0.001 |

`AdamOptimizer` was used to optimize the model.

#### 4. Final Results

My final model results were:
* validation set accuracy of **0.941**
* test set accuracy of **0.916**

### Test a Model on New Images

#### 1. Random German traffic signs found on the web 
| Image | Description |
|:-----:|:-----------:|
|![alt text][image4] | * No Passing * No background noise.   |
|![alt text][image5] | * Bumpy Road * Some background noise behind the sign. |
|![alt text][image6] | * Speed limit (30km/h) * Significant background noise. |
|![alt text][image7] | * Roundabout Mandatory * Some background noise behind the sign. |
|![alt text][image8] | * Road work * Lot of background noise in the image due to presence of a tree. |

These images were resized to `32 x 32 x 3` dimensions. After resizing they look like this:

![alt text][image9]

* **No Passing** sign can be wrongly classified as **End of no passing**
* **Speed limit (30km/h)** can be wrongly classified as **Speed limit (50km/h)**


#### 2. Prediction Results

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| **No Passing**      		| **No Passing**   									| 
| **Bumpy Road**     			| **Road work** 										|
| **Speed limit (30km/h)**					| **Speed limit (30km/h)**											|
| **Roundabout Mandatory**	      		| **Priority road**					 				|
| **Road work**			| **Road work**     							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. 

#### 3. Model Certainty

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.

**No Passing**

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.886        			| **No Passing**  									| 
| 0.060     				| **Roundabout mandatory**										|
| 0.026					| **End of no passing**											|
| 0.026	      			| **Vehicles over 3.5 metric tons prohibited**					 				|
| 0.000001				    | **Right-of-way at the next intersection**      							|

**Bumpy Road**

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.999        			| **Road work**  									| 
| 3.60831564e-07     				| **Bicycles crossing**										|
| 1.72929518e-11					| **Road narrows on the right**											|
| 5.81553374e-14	      			| **Speed limit (80km/h)**					 				|
| 4.31980450e-16				    | **Double curve**      							|

**Speed limit (30km/h)**

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99999881e-01        			| **Speed limit (30km/h)**  									| 
| 1.23033558e-07     				| **Speed limit (20km/h)**										|
| 2.47407927e-15					| **Speed limit (70km/h)**											|
| 3.86806653e-18	      			| **Speed limit (50km/h)**					 				|
| 3.64326159e-18					    | **Roundabout mandatory**      							|

**Roundabout Mandatory**

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00        			| **Priority road**  									| 
| 9.76907000e-10     				| **Roundabout mandatory**										|
| 3.07066317e-10					| **Right-of-way at the next intersection**											|
| 2.45481710e-15	      			| **No passing**					 				|
| 1.69802054e-18					    | **End of no passing**      							|


**Road work**

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.79318619e-01        			| **Road work**  									| 
| 2.06811670e-02     				| **Dangerous curve to the right**										|
| 1.89160417e-07					| **Right-of-way at the next intersection**											|
| 1.68937392e-10	      			| **Road narrows on the right**					 				|
| 3.29088666e-11					    | **Slippery Road**      							|
