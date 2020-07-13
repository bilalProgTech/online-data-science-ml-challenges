# Problem Statement
Here, we need to identify the digit in given images. We have total 70,000 images, out of which 49,000 are part of train images with the label of digit and rest 21,000 images are unlabeled (known as test images). Now, We need to identify the digit for test images. Public and Private split for test images are 40:60 and evaluation metric of this challenge is accuracy. 

# Link Reference
https://datahack.analyticsvidhya.com/contest/practice-problem-identify-the-digits

# Kaggle Link


# Approaches to predict
* 1 CNN Layer - Max Pooling Layer of 16 neurons
* Flatten Layer
* Dense Layer - 1024
* Dense - 512
* Dense - 10 (Output)

# Leaderboard Accuracy
Public LB: 0.9863095238