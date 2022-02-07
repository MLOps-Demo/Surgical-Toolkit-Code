# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 01:19:58 2022

@author: AMIT CHAKRABORTY
"""

# Importing all necessary libraries
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import matplotlib.pyplot as plt
import json 
from sklearn.metrics import confusion_matrix
import yaml

img_width, img_height = 224, 224

params = yaml.safe_load(open("params.yaml"))["training"]

train_data_dir = 'classy/train_data'
validation_data_dir = 'classy/val_data'
nb_train_samples =112
nb_validation_samples = 69
epochs = params["epochs"]
batch_size = params["batch_size"]

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
 
test_datagen = ImageDataGenerator(rescale=1. / 255)
 
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
 
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


def train():
    model = Sequential()
    model.add(Conv2D(32, (2, 2), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))


    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size
    ,verbose=1)
    
    print(model.history.history.keys())
    plt.subplot(1, 2, 1)
    plt.plot(model.history.history['accuracy'])
    plt.plot(model.history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('accuracy.png')
    #plt.show()
    
    plt.subplot(1, 2, 2)
    plt.plot(model.history.history['loss'])
    plt.plot(model.history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    plt.savefig('loss.png')

    #save model file
    model.save_weights('saved-models/model_saved.h5')
    # Reset 
    validation_generator.reset()

    # Evaluate on Validation data
    scores = model.evaluate(validation_generator)
    scores = model.evaluate_generator(validation_generator)
    print("%s%s: %.2f%%" % ("evaluate ",model.metrics_names[1], scores[1]*100))


    print("%s%s: %.2f%%" % ("loss ",model.metrics_names[0], scores[0]))

    with open("scores.json", "w") as fd:
        json.dump({"loss": scores[0], "accuracy": scores[1]}, fd, indent=4)

if __name__ == '__main__':
    train()
