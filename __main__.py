"""Trains a CNN on the MNIST dataset."""


# import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt
# import numpy as np


# import mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# visualize dataset
img_num = 50
plt.imshow(x_train[img_num], cmap='gray')
plt.title(f'Label: {y_train[img_num]}')
plt.show()
