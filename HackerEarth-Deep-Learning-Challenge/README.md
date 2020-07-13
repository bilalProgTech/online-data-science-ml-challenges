# Problem Statement
This International Dance Day, an event management company organized an evening of Indian classical dance performances to celebrate the rich, eloquent, and elegant art of dance. After the event, the company plans to create a microsite to promote and raise awareness among people about these dance forms. However, identifying them from images is a difficult task.

You are appointed as a Machine Learning Engineer for this project. Your task is to build a deep learning model that can help the company classify these images into eight categories of Indian classical dance.

Note

The eight categories of Indian classical dance are as follows:

Manipuri
Bharatanatyam
Odissi
Kathakali
Kathak
Sattriya
Kuchipudi
Mohiniyattam

# Link Reference
https://www.hackerearth.com/problem/machine-learning/identify-the-dance-form-deea77f8/

# Kaggle Link


# Approach to predict
* 3 CNN Layer - Max Pooling Layer of 8 neurons
* 2 CNN Layer - Max Pooling Layer of 16 neurons
* Flatten Layer
* Dropout of 0.01
* Dense Layer - 512
* Dense - 8 (Output)