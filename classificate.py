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
from keras.models import model_from_json

import json

import sys

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
    ###### Load Model ########################################################################################################################################################################################
    
    # load model architeture
    json_file = open("./model/model.json", 'r')
    # read json
    loaded_model_json = json_file.read()
    # close file
    json_file.close()

    # load hyper paramters
    with open("./model/" + args[1]) as json_file:
        hyperparameters = json.load(json_file)

    # create model from json
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("./model/model.h5")
    # compile model
    model.compile(loss = hyperparameters['loss'], optimizer = Adam(learning_rate=hyperparameters['learning rate']), metrics=hyperparameters['metrics'])

    ###### Load Model ########################################################################################################################################################################################
         
    # load data for prediction
    data = pd.read_csv("./data/" + args[2], sep = ";", index_col = 0)
    
    # get values as matrix
    matrix = data.values
    # norazlize matrix
    x_data = normalize(matrix)
    # predict the output
    output = model.predict(x_data)
    # reverse to one dimensional
    output = np.argmax(output, axis=1)

    # classes
    classes = ["Atacante", "Defensor"]
    # applt substitution of output
    output = list( [classes[ value ] for value in output] )
    # add to dataframe
    data["Classe"] = output

    # save
    data.to_csv("./results/" + args[3], sep=";", header=True, index=True)
    print("Saved.")
    


if __name__ == "__main__":
    # get argv
    args = sys.argv
    if len(args) < 2:
        print("Invalid Arguments")
    else:
        main(args)