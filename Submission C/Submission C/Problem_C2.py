# =============================================================================
# PROBLEM C2
#
# Create a classifier for the MNIST Handwritten digit dataset.
# The test will expect it to classify 10 classes.
#
# Don't use lambda layers in your model.
#
# Desired accuracy AND validation_accuracy > 91%
# =============================================================================

import tensorflow as tf


def solution_C2():
    fashion_mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(10000).batch(64)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(64)
    Categories = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    train_images = train_images/255.0
    test_images = test_images/255.0
    from keras_preprocessing.image import ImageDataGenerator
    from tensorflow.keras.optimizers import RMSprop
    
    model = keras.Sequential([
                          keras.layers.Flatten(input_shape=(28, 28)),
                          keras.layers.Dense(128, activation=tf.nn.relu),
                          keras.layers.Dense(10, activation=tf.nn.softmax)
                          ])
    
    model.compile(optimizer=RMSprop(learning_rate=0.001), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
    model.fit(train_images, train_labels, epochs=10)
    model.evaluate(test_images, test_labels)

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    if __name__ == '__main__':
        model = solution_C2()
        model.save("model_C2.h5")