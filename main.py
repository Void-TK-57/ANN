import pandas as pd
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt

def main():
    # load clothes images data
    data = keras.datasets.fashion_mnist
    # get test and train data
    (train_images, train_labels), (test_images, test_labels) = data.load_data()
    # name classes
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
    # preprocces it
    train_images = train_images / 255
    test_images = test_images / 255

    plt.imshow(train_images[7], cmap=plt.cm.Greens)
    plt.show()

    # model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape = (28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])

    # compile model
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # train model
    model.fit(train_images, train_labels, epochs=10)
    
    # evaluate model
    test_loss, test_acc = model.evaluate(test_images, test_labels)

    print("Test Accuracy:", test_acc)

if __name__ == "__main__":
    main()