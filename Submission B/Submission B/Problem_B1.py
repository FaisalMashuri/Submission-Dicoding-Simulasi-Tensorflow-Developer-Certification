# =============================================================================
# PROBLEM B1
#
# Given two arrays, train a neural network model to match the X to the Y.
# Predict the model with new values of X [-2.0, 10.0]
# We provide the model prediction, do not change the code.
#
# The test infrastructure expects a trained model that accepts
# an input shape of [1]
# Do not use lambda layers in your model.
#
# Desired loss (MSE) < 1e-3
# =============================================================================

import numpy as np
import tensorflow as tf
from tensorflow import keras


def solution_B1():
    X = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=float)
    Y = np.array([5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0 ], dtype=float)

    # YOUR CODE HERE
    model = keras.Sequential()
    model.add(keras.layers.Dense(1, input_shape=[1]))
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(X,Y, epochs=1000)
    print(model.predict([-2.0, 10.0]))
    return model

# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    model = solution_B1()
    model.save("model_B1.h5")
