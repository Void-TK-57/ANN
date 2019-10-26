import pandas as pd
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import sys

def show_predict(images, legends):
    
    fig, ax = plt.subplots(7, 7)
    for i in range(7):
        for j in range(7):
            ax[i, j].imshow(images[i + j*7], cmap=cm.Greys)
            ax[i, j].text(5, 0, legends[i + j*7], bbox={'facecolor': 'white', 'pad': 0.2})
    plt.show()

def main(weights = None):
    # load clothes images data
    data = keras.datasets.fashion_mnist
    # get test and train data
    (train_images, train_labels), (test_images, test_labels) = data.load_data()
    # name classes
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
    # preprocces it
    train_images = train_images / 255
    test_images = test_images / 255

    checkpoint_path = "model.ckpt"
    save_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    # model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape = (28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])

    # compile model
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    #model.summary()
    # check path
    if weights is None:
        # then train model
        model.fit(train_images, train_labels, epochs=10, callbacks=[save_callback,])
    else:
        # load model
        model.load_weights(weights)
    print("="*25)
    print(test_images.shape)
    sample_tests = test_images[:49, :, :]
    print(sample_tests.shape)
    # evaluate model
    #test_loss, test_acc = model.evaluate(test_images, test_labels)
    prediction = model.predict(sample_tests)
    indices = [class_names[i] for i in np.argmax(prediction, axis=1)]
    show_predict(sample_tests ,indices)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()