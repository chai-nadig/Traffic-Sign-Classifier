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
| Batch Size     | 256 |
| Epochs         | 15  |
| Learning rate  | 0.001 |

`AdamOptimizer` was used to optimize the model.

#### 4. Final Results

My final model results were:
* validation set accuracy of 0.894 
* test set accuracy of 0.886

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
| **Bumpy Road**     			| **Priority Road** 										|
| **Speed limit (30km/h)**					| **Speed limit (30km/h)**											|
| **Roundabout Mandatory**	      		| **Priority road**					 				|
| **Road work**			| **Road work**     							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. 

#### 3. Model Certainty

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.

**No Passing**

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.884        			| **No Passing**  									| 
| 0.085     				| **Children Crossing**										|
| 0.022					| **Dangerous curve to the right**											|
| 0.003	      			| **Slippery road**					 				|
| 0.0001				    | **Road work**      							|

**Bumpy Road**
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.799        			| **Priority Road**  									| 
| 0.198     				| **No Vehicles**										|
| 0.001					| **Stop**											|
| 0.0000005	      			| **Speed limit (80km/h)**					 				|
| 0.0000000004				    | **Traffic Signals**      							|

**Speed limit (30km/h)**
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.999        			| **Speed limit (30km/h)**  									| 
| 0.000005     				| **Speed limit (50km/h)**										|
| 0.0000000001					| **Speed limit (60km/h)**											|
| 0.00000000001	      			| **Speed limit (20km/h)**					 				|
| 0.000000000001					    | **Speed limit (80km/h)**      							|

**Roundabout Mandatory**
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0        			| **Priority road**  									| 
| 2.0635305e-08     				| **Traffic signals**										|
| 2.0033411e-08					| **Roundabout mandatory**											|
| 1.9728803e-08	      			| **No vehicles**					 				|
| 5.0829407e-09					    | **Children crossing**      							|


**Road work**
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0        			| **Road work**  									| 
| 4.9934885e-16     				| **Bicycles crossing**										|
| 2.1883280e-16					| **Keep right**											|
| 3.1257852e-17	      			| **Bumpy road**					 				|
| 4.4315817e-18					    | **Dangerous curve to the left**      							|
