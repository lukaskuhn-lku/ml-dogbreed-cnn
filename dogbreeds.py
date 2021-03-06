# -*- coding: utf-8 -*-
"""DogBreeds.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1K432CRf5objAIK8IkhgTlj8u6j9dUkqs
"""

#Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import requests
from io import BytesIO
from PIL import Image
keras = tf.keras

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# split the data manually into 80% training, 10% testing, 10% validation
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'stanford_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

get_label_name = metadata.features['label'].int2str  # creates a function object that we can use to get labels

# display 5 images from the dataset
for image, label in raw_train.take(10):
  plt.figure()
  plt.imshow(image)
  plt.title(get_label_name(label))

IMG_SIZE = 160 # All images will be resized to 160x160

def format_example(image, label):
  """
  returns an image that is reshaped to IMG_SIZE
  """
  image = tf.cast(image, tf.float32)
  image = (image/127.5) - 1
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
  return image, label

train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

#Freeze the base model
base_model.trainable = False

#Adding classifier
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = keras.layers.Dense(120, activation="softmax")

#Combine them into one model
model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

model.summary()

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

initial_epochs = 10
validation_steps=20

loss0,accuracy0 = model.evaluate(validation_batches, steps = validation_steps)

history = model.fit(train_batches,
                    epochs=initial_epochs,
                    validation_data=validation_batches)

acc = history.history['accuracy']
print(acc)

predictions = model.predict(test.take(10).batch(10))

for image, label in test.take(10):
  print(get_label_name(label))

for i in range(0,len(predictions)):
  print(get_label_name(np.argmax(predictions[i])))

response = requests.get("https://media.futalis.com/rasseseiten/boston-terrier-rassemerkmale.jpg")
image = Image.open(BytesIO(response.content))
image = tf.cast(np.asarray(image), tf.float32)
image = (image/127.5) - 1
image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))

predictions = model.predict(np.array( [image,] ))

predicted_label = get_label_name(np.argmax(predictions[0]))
predicted_percent = predictions[0][np.argmax(predictions[0])]

print("{:.0%} - {}".format(predicted_percent,predicted_label));

model.save('dog_breed.h5')