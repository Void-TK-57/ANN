# data libraries
import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
import matplotlib.cm as cm

# scikit learn
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# tensorflow keras
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Flatten, MaxPool2D, Conv2D, Dense, Reshape, Dropout
from keras.utils import np_utils, plot_model
from keras.datasets import mnist
from keras.optimizers import SGD, Adam
from keras import losses
from keras.wrappers.scikit_learn import KerasClassifier

import json

import sys

# method to create model
def create_model(hyperparameters):
        # create model
        model = Sequential()
        # add layers
        model.add( Dense( hyperparameters['hidden'][0], input_shape = ( hyperparameters['input'], ), activation = hyperparameters["activation"][0] ) )
        # for every other n_hidden
        for i in range(1, len( hyperparameters['hidden'] )):
            # add layers
            model.add( Dense( hyperparameters['hidden'][i], activation = hyperparameters["activation"][i] ) )
        
        model.add( Dense( hyperparameters['output'], activation = hyperparameters["activation"][-1] ) )
        # compile it
        model.compile(loss = hyperparameters['loss'], optimizer = Adam(learning_rate=hyperparameters['learning rate']), metrics=hyperparameters['metrics'])
        # return model
        return model

# plot model
def plot(model, epochs, ax = plt):
    # x
    x = np.arange(0, epochs)
    # plot loss
    ax.plot(x, model.history["loss"], label="train loss")
    # plot accruacy
    ax.plot(x, model.history["accuracy"], label="train acc")

def normalize(x):
    # function to normalize
    def f(column):
        # get minimum and maximum value
        min_ = np.min(column)
        max_ = np.max(column)
        print(min_)
        print(max_)
        # function to apply the normalization on each value
        g = lambda item: (item - min_)/(max_ - min_)
        # apply to the column
        column = np.apply_along_axis(g , 0, column)
        # return the column
        return column
    # apply f to ach column
    x = np.apply_along_axis(f, 0, x)
    # return x
    return x

    
def main(args):
    # read data
    data = pd.read_csv("./data/" + args[1], sep = ";")
    # load hyper paramters
    with open(args[2]) as json_file:
        hyperparameters = json.load(json_file)
    results = []
    
    # get numpy values
    matrix = data.values
    # get y and x values
    x_data, y_data = matrix[:, :-1].astype(np.int32), matrix[:, -1]
    # norazlize x_data
    print(x_data)
    x_data = normalize(x_data)
    print(x_data)
    return
    
    for train_index,test_index in KFold(6).split(x_data):
        # get train and test
        x_train, x_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]

        # encoder for the class
        encoder = LabelEncoder()
        encoder.fit(matrix[:, -1])
        # transfrom y_train and y_test to numeric
        y_train, y_test = encoder.transform(y_train), encoder.transform(y_test)
        
        # get number of input and output dimensions
        hyperparameters['input'] = x_train.shape[1]
        hyperparameters['output'] = 1 if len(y_train.shape) < 2 else y_train.shape[1]
    
        model=create_model(hyperparameters)
        history = model.fit(x_train, y_train, epochs = hyperparameters['epochs'], batch_size = hyperparameters["batch size"] )
        result = model.evaluate(x_test,y_test, batch_size = hyperparameters["batch size"] )

        result = ['Model evaluation => Loss: ' + str(round( result[0] , hyperparameters['round accuracy']) ) + " , Accuracy: " + str( round( result[1]*100, hyperparameters['round accuracy'] ) ) + "%", model, history, x_test]

        results.append(result)

    for i in range(len(results)):
        print(results[i][1].predict(results[i][3]))
        print(str(i) + ": " + results[i][0] )

    # plot all histories
    fig, axes = plt.subplots(2, 3)
    fig.suptitle("Training Loss and Accuracy")
    history = 0
    # for each ax
    for ax in axes.flatten():
        # call plot
        plot(results[history][2], hyperparameters['epochs'], ax)
        # increase history
        history += 1
    # show plot
    plt.show()

    """
    # create model
    model = create_model(n_input, n_output)

    # train neural network
    history = model.fit(x_train, y_train, epochs = 200)

    # predict
    predictions = model.evaluate(x_test, y_test)
    #plot_model(model, to_file='model.png')
    plot(history, 200)
    """


    

if __name__ == "__main__":
    # get argv
    args = sys.argv
    if len(args) < 2:
        print("Invalid Arguments")
    else:
        main(args)

