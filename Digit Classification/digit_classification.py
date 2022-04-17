# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 00:24:17 2022

@author: inalt
"""

# import libraries


import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense

from keras.optimizers import RMSprop

# load dataset

(x_train, y_train), (x_test, y_test) = mnist.load_data()

g = sns.countplot(y_train)
plt.show()

# data was looked at one by one and no missing data

plt.figure()
plt.imshow(x_train[0], cmap = "gray")
print("label: ",y_train[0])
plt.axis("off")
plt.grid(False)

# normalization

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train /= 255
x_test /= 255

# reshape

img_rows = 28
img_cols = 28

# height = 28px, width = 28px, channal = 1

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

# label encoding

y_train = keras.utils.to_categorical(y_train, num_classes = 10)
y_test = keras.utils.to_categorical(y_test, num_classes = 10)

# CNN Model
# my CNN architechture is in -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = "Same", activation = "relu", input_shape = input_shape))
model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = "same", activation = "relu"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = "same", activation = "relu"))
model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = "same", activation = "relu"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation= "softmax"))

model.summary()

# define the optimizer

optimizer = RMSprop(learning_rate = 0.001)

# compile the model

model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])

epochs = 2
batch_size = 10

history = model.fit(x = x_train, y = y_train, batch_size = batch_size, epochs = epochs, verbose = 1, shuffle = True)

model.save("my_model.h5")

# optimizers = Adadelta => epoch = 25, batch_size = 10 => loss: 0.4611 - accuracy: 0.8548
# optimizers = RMSprop => epoch = 2, batch_size = 10 => loss: 0.1296 - accuracy: 0.9621


 
























