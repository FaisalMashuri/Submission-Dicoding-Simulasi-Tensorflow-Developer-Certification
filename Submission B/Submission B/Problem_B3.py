# ========================================================================================
# PROBLEM B3
#
# Build a CNN based classifier for Rock-Paper-Scissors dataset.
# Your input layer should accept 150x150 with 3 bytes color as the input shape.
# This is unlabeled data, use ImageDataGenerator to automatically label it.
# Don't use lambda layers in your model.
#
# The dataset used in this problem is created by Laurence Moroney (laurencemoroney.com).
#
# Desired accuracy AND validation_accuracy > 83%
# ========================================================================================

import urllib.request
import zipfile
import tensorflow as tf
import os
from keras_preprocessing.image import ImageDataGenerator

    
def solution_B3():
    data_url = 'https://dicodingacademy.blob.core.windows.net/picodiploma/Simulation/machine_learning/rps.zip'
    urllib.request.urlretrieve(data_url, 'rps.zip')
    local_file = 'rps.zip'
    zip_ref = zipfile.ZipFile(local_file, 'r')
    zip_ref.extractall('data/')
    zip_ref.close()


    TRAINING_DIR = "data/rps/"
    datagen = ImageDataGenerator(rescale=1./255,
                             horizontal_flip=True,
                             zoom_range=0.2,
                             shear_range=0.2,
                             rotation_range=20,
                             validation_split=0.2)
        # YOUR CODE HERE)

    train_generator = datagen.flow_from_directory(TRAINING_DIR,
                                              target_size=(150,150),
                                              color_mode='rgb',
                                              class_mode='categorical',
                                              subset='training') # YOUR CODE HERE

    valid_generator = datagen.flow_from_directory(TRAINING_DIR,
                                              target_size=(150,150),
                                              color_mode='rgb',
                                              class_mode='categorical',
                                              subset='validation')

    # YOUR CODE HERE, end with 3 Neuron Dense, activated by softmax
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(8, (3,3), input_shape=(150,150,3)))
    model.add(tf.keras.layers.MaxPooling2D(2,2))

    model.add(tf.keras.layers.Conv2D(16, (3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(2,2))

    model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(2,2))

    model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(2,2))

    model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(2,2))


    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(train_generator, epochs=10,
                        validation_data=valid_generator,
                        verbose=1)
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    model = solution_B3()
    model.save("model_B3.h5")

