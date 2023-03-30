# cv-pytorch
Project Name: Pytorch Linear Regression and CNN Models
Overview:
In this project, I have implemented linear regression and convolutional neural network (CNN) models using the Pytorch deep learning framework. I have started by building a simple linear regression model, followed by a CNN model on FashionMNIST dataset. Finally, I have implemented a CNN model to classify images of sushi, pizza, and steak.

Requirements:
Python (version >= 3.6)
Pytorch (version >= 1.8.1)
Torchvision (version >= 0.9.1)
NumPy (version >= 1.19.5)
Matplotlib (version >= 3.3.4)
Project Structure:
linear_regression.py - Python script containing the implementation of the linear regression model.
cnn_fashion_mnist.py - Python script containing the implementation of the CNN model on FashionMNIST dataset.
cnn_sushi_pizza_steak.py - Python script containing the implementation of the CNN model to classify images of sushi, pizza, and steak.
data folder - contains the training and testing data for the CNN models.
README.md - ReadMe file for the project.
Linear Regression Model:
We have implemented a simple linear regression model using Pytorch. The model takes a single input variable and predicts the output based on a linear combination of the input variable. The model is trained on a randomly generated dataset and the mean squared error (MSE) is used as the loss function. 

CNN Model on FashionMNIST Dataset:
We have implemented a CNN model on the FashionMNIST dataset using Pytorch. The dataset contains 60,000 training images and 10,000 testing images of 10 different fashion categories. The model consists of two convolutional layers followed by two fully connected layers. The cross-entropy loss function is used for training the model. 

CNN Model to Classify Images of Sushi, Pizza, and Steak:
We have implemented a CNN model to classify images of sushi, pizza, and steak using Pytorch. The dataset consists of 1500 training images and 450 testing images of the three food categories. The model consists of three convolutional layers followed by two fully connected layers. The cross-entropy loss function is used for training the model. 




