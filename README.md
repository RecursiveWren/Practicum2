# Practicum2
German Traffic Sign Classifier with Keras, TensorFlow, and Jupyter Notebook

## Project Summary
The goal for this project was to create an immage classification model that could correctly recognize German traffic signsbased on a collection of labeled images gathered by the Institut Fur Neuroinformatik:http://benchmark.ini.rub.de/?section=home&subsection=news     
     
The underlying motivation for taking on this project was:     
  * Gain experience working with image processing in Python
  * Build my first convolutional neural network
  * Compare model performance between 3 different optimization parameters.
  
      
## Implementation Details and Dependencies     
This project was completed using Python's deep learning library, Keras, with a TensorFlow backend. Versions and libraries used:    
* Anaconda 3    
* Python 3.5    
* Jupyter Notebook    
* Keras

# Data Collection, Cleaning and EDA    
The dataset consisted of over 50,000 images saved as .p files. The data was broken down into 3 subsets: 1) train.p 2) test.p and 3) valid.p. Upon inspection of the training images, it became apparent that this collection had already been pre-cropped and all traffic signs centered. The image quality varied greatly between each image, with some signs appearing very dark, or otherwise obscured due to rain/fog. Atmospheric differences in the photes can greatly impact the performance of a model, if not properly addressed. Inspecting the pixel values of each image (and also taking summary stats such as mean, min and max), I could see that there were large descrepencies between the images, and they would need to be normalized before attempting to build the models. As a last step for EDA, I inspected the distribution of type of sign within the training set and the test set. While there were differences in the total nmber of labels for each sign, this seemed to be aceptable for the model I was attempting to build. Both the training set and the test set contained roughly the same mix of labels, and each of the 43 labels were represented.    
# Building the Models    
All models were sequential, and contained the same parameters, with the exception of the optimizer. I wanted to test the impact that the optimizer made to the model's performance, so to achieve this, I opted to only change the optimizers. The three optimizers I tested for this project were:    
## SGD (Stochastic Gradient Decent)    
Stochastic Gradient Decent optimizer offers good support for momentum, learning rate decay, and Nesterov momentum.    
## Adagrad    
Adagrad is an adaptive subgradient optimizer for stochastic optimization. See reference: http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf     
## Adam     
Adam is another method which offeres stochastic optimization via a first order gradient based algorithm. See reference: https://arxiv.org/abs/1412.6980v8    

# Results 
The Adam model performs better than other models, showing better accuracy in training and validation as well as significantly less loss at each epoch.
    
    Overlap Comparison
![alt tag](https://github.com/RecursiveWren/Practicum2/blob/master/Practicum%20Graphs.png)
    Individual Models   
![alt tag](https://github.com/RecursiveWren/Practicum2/blob/master/Practicum%20graphs2.png)

# Next Steps
* Create confusion matrix in Python, using the axis as the images rather than the label numbers, to be able to determine which images are most commonly mis-classified.    
* Work further with model train to achieve higher accuracy  
