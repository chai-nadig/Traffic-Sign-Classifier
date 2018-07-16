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
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


