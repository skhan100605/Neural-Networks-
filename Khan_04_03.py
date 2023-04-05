# Khan, Sahiba
# 1002_083_293
# 2022_11_13
# Assignment_04_03
import pytest
import numpy as np
from cnn import CNN
import os
import tensorflow
from tensorflow.keras.datasets import cifar10
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,MaxPool2D,InputLayer,Flatten
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import sparse_categorical_crossentropy,hinge,mean_squared_error
from tensorflow.keras.optimizers import Adam,RMSprop,SGD,Adagrad
from numpy import loadtxt

def test_train_evaluate():
    print('inside')
    # define the keras model
    # load the dataset
    dataset = loadtxt('Covid-Live2.txt', delimiter=',')
    # split into input (X) and output (y) variables
    X = dataset[:,0:8]
    y = dataset[:,8]
    # define the keras model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile the keras model
    model.compile(loss='mean_squared_error', optimizer="SGD", metrics=['accuracy'])
    # fit the keras model on the dataset
    model.fit(X, y, epochs=150, batch_size=10)
    # evaluate the keras model
    _, accuracy = model.evaluate(X, y)
    # print('Accuracy: %.2f' % (accuracy*100))
    my_cnn = CNN()
    my_cnn.add_input_layer(shape = (8,))
    my_cnn.append_dense_layer(num_nodes=12,activation="relu",name="dense1")
    my_cnn.append_dense_layer(num_nodes=8,activation="relu",name="dense2")
    my_cnn.append_dense_layer(num_nodes=1,activation="sigmoid",name="dense3")
    my_cnn.set_loss_function(loss = "mean_squared_error")
    my_cnn.set_optimizer(optimizer = "SGD")
    my_cnn.set_metric(metric = "accuracy")
    my_cnn.train(X_train = X, y_train = y, num_epochs = 10, batch_size = 10)
    acc = my_cnn.evaluate(X = X, y = y)
    assert (round(acc[1],1) == round(accuracy,1))
