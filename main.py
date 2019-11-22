# data libraries
import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import seaborn as sns; sns.set()

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
        # add first layer
        model.add( Dense( hyperparameters['layers'][1], input_shape = ( hyperparameters['layers'][0], ), activation = hyperparameters["activation"][0] ) )
        # for every other layer
        for i in range(2, len( hyperparameters['layers'] )):
            # add layers
            model.add( Dense( hyperparameters['layers'][i], activation = hyperparameters["activation"][i-1] ) )
        # compile it
        model.compile(loss = hyperparameters['loss'], optimizer = Adam(learning_rate=hyperparameters['learning rate']), metrics=hyperparameters['metrics'])
        # return model
        return model

# plot model
def plot(model, epochs, ax = plt):
    # x
    x = np.arange(0, epochs)
    # plot accruacy
    ax.plot(x, model.history["accuracy"], label="train acc")

# save model
def save(model, file = './model/model.json'):
    # serialize model to JSON
    model_json = model.to_json()
    # open file
    with open(file, "w") as json_file:
        json_file.write(model_json)

def normalize(x):
    # function to normalize
    def f(column):
        # get minimum and maximum value
        mean_ = np.mean(column)
        std_ = np.std(column)
        # function to apply the normalization on each value
        g = lambda item: (item - mean_)/std_
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
    data = pd.read_csv("./data/" + args[2], sep = ";", index_col = 0)

    # load hyper paramters
    with open("./model/" + args[1]) as json_file:
        hyperparameters = json.load(json_file)
    
    # get values as matrix
    matrix = data.values
    # get y and x values
    x_data, y_data = matrix[:, :-1].astype(np.float64), matrix[:, -1]
    # norazlize x_data
    x_data = normalize(x_data)
    # get train and test
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size = hyperparameters["training slice"])

    # encoder for the class
    encoder = LabelEncoder()
    encoder.fit(matrix[:, -1])
    # transfrom y_train and y_test to numeric
    y_train, y_test = encoder.transform(y_train), encoder.transform(y_test)
    
    # create model
    model=create_model(hyperparameters)
    # train model
    history = model.fit(x_train, y_train, epochs = hyperparameters['epochs'], batch_size = hyperparameters["batch size"] )
    # evaluate model
    result = model.evaluate(x_test,y_test, batch_size = hyperparameters["batch size"] )

    # print results
    print('Model evaluation => Loss: ' + str( result[0] ) + " , Accuracy: " + str( result[1]*100 ) + "%")
    
    # plot accuracy
    plt.plot(history.history["accuracy"])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    # show plot
    plt.show()

    ###### Save Model ########################################################################################################################################################################################
    # save model
    save(model)
    # save weights
    model.save_weights("./model/model.h5")
    ###### Save Model ########################################################################################################################################################################################
    

if __name__ == "__main__":
    # get argv
    args = sys.argv
    if len(args) < 2:
        print("Invalid Arguments")
    else:
        main(args)

