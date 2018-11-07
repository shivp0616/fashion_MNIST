# -*- coding: utf-8 -*-
"""
@author: shivp1606
"""
#inporting the necessary libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


#Import the Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()


#As the class names are not included with the dataset, store them here to use later
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


#The pixel values of the images falls in range of 0-255
#We are converting it in range 0-1
X_train = X_train / 255.0
X_test = X_test / 255.0


'''
#Display the first 5 images from the training set and display the class name
plt.figure(figsize=(10,10))
for i in range(5):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
'''

#build the model
'''Flatten layer transforms the format of the images from a 2d-array (of 28 by 28 pixels)
   to a 1d-array of 28 * 28 = 784 pixels.
   Dense layer are densely-connected, or fully-connected, neural layers.
   
   In the last layer, we have taken the units as 10 because there are 10 different classes in the target.
   Each unit will be responsible for their respective class. They will generate a probablity and the one having the highest will be selected.
'''
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


#Before the model is ready for training, it needs a few more settings. These are added during the model's compile step
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#Training the model
model.fit(X_train, y_train, epochs=10)


#Evaluate accuracy
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

#you can also predict the class of some new image
#but note that the image should also be in the same format and size as of those on which we have trained.
#for that we can use model.predict() method
predictions = model.predict(X_test[0])
predictions[0]
np.argmax(predictions[0])
