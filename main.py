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
from keras.utils import np_utils, plot_model, to_categorical
from keras.datasets import mnist
from keras.optimizers import SGD, Adam
from keras import losses
from keras.wrappers.scikit_learn import KerasClassifier
from keras.metrics import CategoricalAccuracy

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
        model.compile(loss = hyperparameters['loss'], optimizer = Adam(learning_rate=hyperparameters['learning rate']), metrics=hyperparameters['metrics'] )
        # return model
        return model

# plot model
def plot(history, epochs, ax = plt):
    # x
    x = np.arange(0, epochs)
    print(history.history.keys())
    # plot accruacy
    ax.plot(x, history.history["accuracy"], label="train acc")() 

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

    # data slices
    training_slice = hyperparameters["training slice"]
    validation_slice = hyperparameters["validation slice"]
    test_slice = 1.0 - training_slice - validation_slice
    # from data get train and rest
    x_train, x_rest, y_train, y_rest = train_test_split(x_data, y_data, train_size = training_slice)
    # from rest get validation and test
    x_validation, x_test, y_validation, y_test = train_test_split(x_rest, y_rest, train_size = validation_slice/(validation_slice + test_slice) )

    # encoder for the class
    encoder = LabelEncoder()
    encoder.fit(matrix[:, -1])
    # transfrom y_train and y_test to numeric
    y_train, y_test, y_validation = encoder.transform(y_train), encoder.transform(y_test), encoder.transform(y_validation)
    # transform to categorical
    y_train, y_test, y_validation = to_categorical(y_train, len(encoder.classes_)), to_categorical(y_test, len(encoder.classes_)), to_categorical(y_validation, len(encoder.classes_))
    
    # create model
    model=create_model(hyperparameters)
    # train model
    history = model.fit(x_train, y_train, epochs = hyperparameters['epochs'], batch_size = hyperparameters["batch size"], validation_data = (x_validation , y_validation) )
    # evaluate model
    result = model.evaluate(x_test,y_test, batch_size = hyperparameters["batch size"] )

    # print results
    print('Model evaluation => Loss: ' + str( result[0] ) + 'Accuracy: ' + str( result[1]*100 ) + '%')

    # plot accuracy
    fig, axes = plt.subplots(3, 1)
    # plot accuracy
    axes[0].plot(history.history["accuracy"], color="blue")
    axes[0].plot(history.history["val_accuracy"], color="green")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(['train', 'validation'], loc='lower right')
    # plot categorical accuracy
    axes[1].plot(history.history["categorical_accuracy"], color="blue")
    axes[1].plot(history.history["val_categorical_accuracy"], color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Categorical Accuracy")
    axes[1].legend(['train', 'validation'], loc='lower right')
    # plot loss
    axes[2].plot(history.history["loss"], color="blue")
    axes[2].plot(history.history["val_loss"], color="green")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].legend(['train', 'validation'], loc='upper right')
    # show plot
    plt.savefig('./results/results.png')
    plt.show()

    ###### Save Model ########################################################################################################################################################################################
    # save model
    save(model)
    # save weights
    model.save_weights("./model/model.h5")
    print("Model Saved.")
    ###### Save Model ########################################################################################################################################################################################
    

if __name__ == "__main__":
    # get argv
    args = sys.argv
    if len(args) < 2:
        print("Invalid Arguments")
    else:
        main(args)

