import pandas as pd
import numpy as np 
import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import sys

def main(args):
    # read data
    data = pd.read_csv("./data/" + args[1])
    print(list(data.columns))
    
    # Neural Network
    n_input = 2
    n_hidden = 3
    n_output = 2

    # input
    X = tf.placeholder(tf,float32, shape = (None, n_input), name = "X")
    Y = tf.placeholder(tf,int64, shape = (None), name = "Y")

def neuron_layer(X, n_neuros, name, activation = None):
    with tf.name_scope(name):
        # get inputs
        n_inputs = int(X.get_shape()[1])
        # set standard derivation to the gaussian to 
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="weights")
        b = tf.Variable(tf.zeros([n_neurons]), name="biases")
        z = tf.matmul(X, W) + b
        if activation=="relu":
            return tf.nn.relu(z)
        else:
            return z

    

if __name__ == "__main__":
    # get argv
    args = sys.argv
    if len(args) < 2:
        print("Invalid Arguments")
    else:
        main(args)

