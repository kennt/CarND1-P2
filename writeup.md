#**Traffic Sign Recognition** 

---


##**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image0]: ./train0.png "Training image 0"
[image1]: ./train0n.png "Normalized Training image 0"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./images-1.png "Traffic Sign 1"
[image5]: ./images-2.png "Traffic Sign 2"
[image6]: ./images-3.png "Traffic Sign 3"
[image7]: ./images-4.png "Traffic Sign 4"
[image8]: ./images-5.png "Traffic Sign 5"
[image9]: ./histogram-training.png "Training Set Histogram"
[image10]: ./histogram-validation.png "Validation Set Histogram"
[image11]: ./histogram-test.png "Test Set Histogram"
[image12]: ./model.png "Model architecture diagram"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/kennt/CarND-P2/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Data Set Summary

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is **34799**
* The size of the validation set is **4410**
* The size of test set is **12630**
* The shape of a traffic sign image is **32x32x3**
* The number of unique classes/labels in the data set is **43**

####2. Exploratory Data Set Visualizations

Here is an exploratory visualization of the data set. I have created three histograms (bar charts) of the Training, Validation, and Test sets.

The distributions are very similar, having the same peaks and dips.  Another
interesting point is that the peaks have ~10x more samples than the dips.

![Training Set][image9]
![Validation Set][image10]
![Test Set][image11]

###Design and Test a Model Architecture

####1. Data Preprocessing

Due to some favorable experiences with the previous project, I tried converting the data to HSV rather than use RGB.  I found it made no difference to the accuracy.  As a second try, I pulled just the V channel (so used a 32x32x1 image) and still found no difference.  So I just kept the RGB image.

If it truly made no difference, I think I should have just kept the 32x32x1 image, since it would have reduced the number of parameters needed.

At the end I normalized the data, from [0,255] to (-1, +1).

Here is a sample image

![Training image][image0]

Here is the same sample image (normalized)

![Normalized training image][image1]



####2. Model Architecture

This is a diagram of the final model.  This model was based off of the LeNet example.


![Model architecture diagram][image12]
 

####3. Training of the model

**Hyperparameters**

* Number of Epochs = **15**
* Batch Size = **128**
* Learning Rate = **0.001**
* Dropout Rate = **0.5**
* Biases/Weights starting mean = **0**
* Biases/Weights starting standard deviations dependent on input sizes

The optimizer is the same optimizer used by the LeNet example, the AdamOptimizer.

####4. Problem approach

**Final model results**

* Training set accuracy = **0.997**
* Validation set accuracy = **0.961**
* Test set accuracy = **0.947**


Initially, I started out with the model from the LeNet example (due to that they were both involve image recognition). I modified the code for the problem set (increased the final output to 43 from 10).  This gave me ~0.89 accuracy, which wasn't enough.

At this stage, I played with the learning rate and batch size to find a good value. It turns out that higher learning rates would show higher volatility and not settle down.  Also the batch size seemed to do well enough (seemed to be a good tradeoff of processing time vs. accuracy).

After settling on some of the hyperparameters, the first model modification I made was to add a dropout layer.  This comes after the first fully connected layer.  This made a pretty large increase in accuracy to ~0.94.

After that I made some additional modifications (doubled the depths of the convolution layers and changing the initial values of the biases and weights). These modification increased the performance a little (up to ~0.96), but the main advantage is that they seemed to converge more quickly.


###Test a Model on New Images

####1. Five new German traffic signs

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first one may be difficult since the diagonal lines are used in other signs (although they are over a background image). Some of the other images also have less training samples.

The images I found were all pretty clear and bright.  It would have been interesting to find pictures of images taken at night (since they would look different, especially since I did not do grayscale conversion).


####2. Prediction results on the five new traffic signs

Here are the results of the prediction:

| Image			        |     Prediction	        | 
|:---------------------:|:---------------------------------------------:| 
| End of all speed and passing limits (32)	| End of all speed and passing limit (32)   | 
| Bumpy road (22)    						| Bumpy road (22)                      |
| Right-of-way at the next intersection (11) | Right-of-way at the next intersection (11) |
| No passing (9)	   						| No passing (9)                       |
| Roundabout mandatory (40)					| Roundabout mandatory (40)            |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%, which is better than the test set's accuracy of 94.7%.


####3. Top 5 softmax probabilities

The code for making predictions on my final model is located in the 30th cell of the Ipython notebook.

--------

For the first image, the model is very sure that this is an end to all speed and passing limits sign (probability of 0.98), and the image does contain an end to all speed and passing limits sign. The top five soft max probabilities were

| Probability         	|     Prediction	       | 
|:---------------------:|:---------------------------------------------:| 
| .9822        			| End of all speed and passing limits (32) | 
| .0164     			| End of speed limit (80km/h) (6)          |
| .0015					| End of no passing (41)                   |
| .0002	      			| End of no passing by vehicles over 3.5 metric tons (42) |
| .00002			    | Turn left ahead (34)                     |

--------

For the second image, the model is very sure that this is a bumpy road sign (probability of 1.0), and the image does contain a bumpy road sign. The top five soft max probabilities were

| Probability         	|     Prediction	       | 
|:---------------------:|:---------------------------------------------:| 
| 1.0000       			| Bumpy road (22)                               | 
| 6.4e-11     			| Bicycles crossing (29)                        |
| 5.7e-12				| Traffic signals (26)                          |
| 4.5e-12    			| Road narrows on the right (25)                |
| 1.1e-15			    | Dangerous curve to the right (20)             |

--------

For the third image, the model is very sure that this is a right-of-way at the next intersection sign (probability of 1.0), and the image does contain a right-of-way at the next intersection sign. The top five soft max probabilities were

| Probability         	|     Prediction	       | 
|:---------------------:|:---------------------------------------------:| 
| 1.0000       			| Right-of-way at the next intersection (11)     |
| 6.2e-20     			| Beware of ice/snow (30)                        |
| 1.7e-21				| Pedestrians (27)                               |
| 1.9e-29    			| Double curve (21)                              |
| 9.6e-32			    | Roundabout mandatory (40)                      |

--------

For the fourth image, the model is very sure that this is a no passing sign (probability of 1.0), and the image does contain a no passing sign. The top five soft max probabilities were

| Probability         	|     Prediction	       | 
|:---------------------:|:---------------------------------------------:| 
| 1.0000       			| No passing (9)     	                         |
| 4.5e-22     			| No vehicles (15)                              |
| 1.1e-34				| Speed limit (50km/h) (3)                      |
| 4.0e-35    			| Vehicles over 3.5 metric tons prohibited (16) |
| 2.8e-35			    | Dangerous curve to the left (19)              |

--------

For the fifth image, the model is very sure that this is a roundabout mandatory sign (probability of 0.9999), and the image does contain a roundabout mandatory sign. The top five soft max probabilities were

| Probability         	|     Prediction	       | 
|:---------------------:|:---------------------------------------------:| 
| 1.0000       			| Roundabout mandatory (40)                     |
| 1.8e-07     			| Right-of-way at the next intersection (11)    |
| 1.6e-09				| General caution (18)                          |
| 2.9e-12    			| Speed limit (100km/h) (7)                     |
| 2.3e-12			    | Pedestrians (27)                              |



