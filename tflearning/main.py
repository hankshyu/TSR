from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from stn import spatial_transformer_network as transformer
import tensorflow.keras.activations as ac
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2

train = ImageDataGenerator(rescale=1 / 255,validation_split=0.2)
test = ImageDataGenerator(rescale=1 / 255)

train_dataset = train.flow_from_directory("/Users/xuzihan/Desktop/TSRD/tsrdDatabasefull/train",
                                          target_size=(150, 150),
                                          class_mode="categorical",
                                          subset="training")
valid_dataset = train.flow_from_directory("/Users/xuzihan/Desktop/TSRD/tsrdDatabasefull/train",
                                          target_size=(150, 150),
                                          class_mode="categorical",
                                          subset="validation")

test_dataset = test.flow_from_directory("/Users/xuzihan/Desktop/TSRD/tsrdDatabasefull/test",
                                        target_size=(150, 150),
                                        class_mode="categorical")

batch_size = 100
model = tf.keras.models.Sequential(
    [
        tf.keras.Input(shape=(150, 150, 3)),
        tf.keras.layers.Conv2D(32, kernel_size=(5, 5)),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Conv2D(64, kernel_size=(5, 5)),
        tf.keras.layers.ReLU(),

        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),


        tf.keras.layers.Conv2D(64, kernel_size=(7, 7), padding="same",activation="relu"),
        tf.keras.layers.Conv2D(64, kernel_size=(7, 7), padding="same",activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256),
        tf.keras.layers.LeakyReLU(alpha=0.3),
        tf.keras.layers.Dense(58, activation="softmax")
    ]
)
model.summary()
'''
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(train_dataset, steps_per_epoch=20,validation_data=valid_dataset,batch_size=20, epochs=10)

score = model.evaluate(test_dataset, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
print("hello this is the end ***")
'''