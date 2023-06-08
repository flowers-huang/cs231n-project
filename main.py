import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import numpy as np
import coremltools as ct
import pathlib
import datetime
import os
print("Finished imports")

##### VERSION CONTROL
print(f"Tensor Flow Version: {tf.__version__}")
print(f"numpy Version: {np.version.version}")

##### INPUT IMPORT SANITY CHECK
data_dir = pathlib.Path("input/train")
os.listdir(data_dir)
image_count = len(list(data_dir.glob('*/*.png')))
print("Total # Images in Input Dataset: {}".format(image_count))
# classnames in the dataset specified
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt" ])
print("All Emotion Classes:")
print(CLASS_NAMES)
# print length of class names
output_class_units = len(CLASS_NAMES)
print("Total # of Classes in Input Dataset: {}".format(output_class_units))

NUM_EPOCHS = 1

##### MODEL

model = tf.keras.models.Sequential([
    # 1st conv
  tf.keras.layers.Conv2D(96, (11,11),strides=(4,4), activation='relu', input_shape=(227, 227, 3)),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPooling2D(2, strides=(2,2)),
    # 2nd conv
  tf.keras.layers.Conv2D(256, (11,11),strides=(1,1), activation='relu',padding="same"),
  tf.keras.layers.BatchNormalization(),
     # 3rd conv
  tf.keras.layers.Conv2D(384, (3,3),strides=(1,1), activation='relu',padding="same"),
  tf.keras.layers.BatchNormalization(),
    # 4th conv
  tf.keras.layers.Conv2D(384, (3,3),strides=(1,1), activation='relu',padding="same"),
  tf.keras.layers.BatchNormalization(),
    # 5th Conv
  tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu',padding="same"),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPooling2D(2, strides=(2, 2)),
  # To Flatten layer
  tf.keras.layers.Flatten(),
  # To FC layer 1
  tf.keras.layers.Dense(4096, activation='relu'),
    # add dropout 0.5 ==> tf.keras.layers.Dropout(0.5),
  #To FC layer 2
  tf.keras.layers.Dense(4096, activation='relu'),
    # add dropout 0.5 ==> tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(output_class_units, activation='sigmoid')
])

data_dir2 = pathlib.Path("input/test")

BATCH_SIZE = 32             # Can be of size 2^n, but not restricted to. for the better utilization of memory
IMG_HEIGHT = 227            # input Shape required by the model
IMG_WIDTH = 227             # input Shape required by the model
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

# Rescalingthe pixel values from 0~255 to 0~1 For RGB Channels of the image.
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
# training_data for model training
train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH), #Resizing the raw dataset
                                                     classes = list(CLASS_NAMES))

val_data_gen = image_generator.flow_from_directory(directory=str(data_dir2),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH), #Resizing the raw dataset
                                                     classes = list(CLASS_NAMES))

model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy',tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.SensitivityAtSpecificity(0.5), tf.keras.metrics.SpecificityAtSensitivity(0.5), tf.keras.metrics.AUC(curve='ROC')])

# Summarizing the model architecture and printing it out
model.summary()

import time
start = time.time()

history = model.fit(
      train_data_gen,
      steps_per_epoch = STEPS_PER_EPOCH,
      epochs= NUM_EPOCHS,
      validation_data=val_data_gen
)

# Saving the model
model.save('AlexNet_saved_model/')
print("Total time: ", time.time() - start, "seconds")

print("Beginning Conversion")
#model_from_tf = ct.convert(model, convert_to="mlprogram")
mlmodel = ct.convert('AlexNet_save_model', convert_to="mlprogram")

model_from_tf.save('model.mlmodel')
