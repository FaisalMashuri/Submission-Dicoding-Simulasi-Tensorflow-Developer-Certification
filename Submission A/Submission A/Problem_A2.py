# =====================================================================================
# PROBLEM A2 
#
# Build a Neural Network Model for Horse or Human Dataset.
# The test will expect it to classify binary classes. 
# Your input layer should accept 150x150 with 3 bytes color as the input shape.
# Don't use lambda layers in your model.
#
# The dataset used in this problem is created by Laurence Moroney (laurencemoroney.com).
#
# Desired accuracy and validation_accuracy > 83%
# ======================================================================================

import urllib.request
import zipfile
import tensorflow as tf
import os
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense

def solution_A2():
    data_url_1 = 'https://dicodingacademy.blob.core.windows.net/picodiploma/Simulation/machine_learning/horse-or-human.zip'
    urllib.request.urlretrieve(data_url_1, 'horse-or-human.zip')
    local_file = 'horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_file, 'r')
    zip_ref.extractall('data/horse-or-human')

    data_url_2 = 'https://dicodingacademy.blob.core.windows.net/picodiploma/Simulation/machine_learning/validation-horse-or-human.zip'
    urllib.request.urlretrieve(data_url_2, 'validation-horse-or-human.zip')
    local_file = 'validation-horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_file, 'r')
    zip_ref.extractall('data/validation-horse-or-human')
    zip_ref.close()


    TRAINING_DIR = "/content/data/horse-or-human"
    datagen = ImageDataGenerator(rescale=1./255,
                                   horizontal_flip=True,
                                   zoom_range=0.2,
                                   shear_range=0.2,
                                   rotation_range=10,
                                   fill_mode='nearest',
                                   validation_split=0.2
                                   )
        # YOUR CODE HERE)

    
    train_generator = datagen.flow_from_directory(TRAINING_DIR,
                                                                        target_size=(150, 150),
                                                                        color_mode='rgb',
                                                                        class_mode='binary',
                                                                        batch_size=32,
                                                                        shuffle=True,
                                                                        subset='training')

    valid_generator = datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(150, 150),
        class_mode='binary',
        subset='validation')
     # YOUR CODE HERE

    model = Sequential()
    model.add(Conv2D(8, (3,3), activation='relu', input_shape=(150,150,3)))
    model.add(Conv2D(8, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(16, (3,3), activation='relu'))
    model.add(Conv2D(16, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_generator, validation_data=valid_generator, epochs=20, verbose=2)

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    model = solution_A2()
    model.save("model_A2.h5")