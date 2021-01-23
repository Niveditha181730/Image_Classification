# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 22:55:33 2020

@author: 13657
"""
#%%
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv
import os

#%% the path of the folder may need to be changed in order to fit your computer
path_train = ['train\\'+i for i in os.listdir('train') if 'jpg' in i] 
path_test = ['test\\'+i for i in os.listdir('test') if 'jpg' in i] 
path = np.append(path_train, path_test)

# Basic Information of photos
df_picSize = pd.DataFrame([],columns=['width','height','rgb','category','path','train-test'])
for i in path:
    temp_i = pd.DataFrame([np.append(np.array(cv.imread(i).shape),[(i.split('\\')[1]).split('_')[0],i,i.split('\\')[0]])],columns=['width', 'height', 'rgb','category','path','train-test'])
    df_picSize = df_picSize.append(temp_i)
del i,temp_i,path_train,path_test,path

# Formatting dataset
x_train = np.array([cv.resize(cv.imread(path),(150,150)) for path in df_picSize['path'][df_picSize['train-test']=='train']])
y_train = np.array([category for category in df_picSize['category'][df_picSize['train-test']=='train']])
x_test = np.array([cv.resize(cv.imread(path),(150,150)) for path in df_picSize['path'][df_picSize['train-test']=='test']])
y_test = np.array([category for category in df_picSize['category'][df_picSize['train-test']=='test']])  

# x_data processing: pixel standerization
x_train = x_train / 255
x_test = x_test / 255

# y_data processing: one hot encoding label
fruitDict = {
    'apple':0,
    'banana':1,
    'orange':2,
    'mixed':3
    }
y_train = np.array([fruitDict[i] for i in y_train],dtype = np.uint8)
y_test = np.array([fruitDict[i] for i in y_test],dtype = np.uint8)
# one hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 4)
y_test = tf.keras.utils.to_categorical(y_test, 4)

#%%  Model_1 (Team 9's best model architecture)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

#%%  Model_2 (Triple convolutional layers)
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
# model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
# model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
# model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
# model.add(tf.keras.layers.Flatten())	
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(4, activation='softmax'))
# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# model.summary()

#%%
history = model.fit(x_train, 
                    y_train,
                    batch_size = 20 ,
                    epochs = 20,
                    verbose = 1,
                    validation_data = (x_test, y_test)) 

#%% Plot
def fig_loss(history):
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, 'b', label='Training loss')
    plt.plot(epochs, val_loss_values, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

def fig_acc(history):
    history_dict = history.history 
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()

#%%
fig_loss(history)
fig_acc(history)

#%%

# model.predict(x_test)
# print("score =", model.evaluate(x_test, y_test))



