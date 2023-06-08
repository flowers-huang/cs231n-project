import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import numpy as np
import coremltools as ct
import pathlib
import datetime
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import confusion_matrix , classification_report 
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score

from IPython.display import clear_output
import warnings
warnings.filterwarnings('ignore')
print("Finished imports")

##### VERSION CONTROL
print(f"Tensor Flow Version: {tf.__version__}")
print(f"numpy Version: {np.version.version}")

##### INPUT IMPORT SANITY CHECK
train_dir = pathlib.Path("input/train")
test_dir = pathlib.Path("input/test")

SEED = 12
IMG_HEIGHT = 48
IMG_WIDTH = 48
BATCH_SIZE = 64
BATCHSIZES = [64]
EPOCHS = 30
FINE_TUNING_EPOCHS = 20
## ORIGINALLY 30 and 20 for EPOCHS
LR = 0.01
NUM_CLASSES = 7
EARLY_STOPPING_CRITERIA=3
CLASS_LABELS  = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', "Surprise"]

# DATA PREPROCESSING

for batch in BATCHSIZES:
    BATCH_SIZE = batch


    preprocess_fun = tf.keras.applications.densenet.preprocess_input

    train_datagen = ImageDataGenerator(horizontal_flip=True,
                                       width_shift_range=0.1,
                                       height_shift_range=0.05,
                                       rescale = 1./255,
                                       validation_split = 0.2,
                                       preprocessing_function=preprocess_fun
                                      )
    test_datagen = ImageDataGenerator(rescale = 1./255,
                                      validation_split = 0.2,
                                      preprocessing_function=preprocess_fun)

    train_generator = train_datagen.flow_from_directory(directory = train_dir,
                                                        target_size = (IMG_HEIGHT ,IMG_WIDTH),
                                                        batch_size = BATCH_SIZE,
                                                        shuffle  = True , 
                                                        color_mode = "rgb",
                                                        class_mode = "categorical",
                                                        subset = "training",
                                                        seed = 12
                                                       )

    validation_generator = test_datagen.flow_from_directory(directory = train_dir,
                                                             target_size = (IMG_HEIGHT ,IMG_WIDTH),
                                                             batch_size = BATCH_SIZE,
                                                             shuffle  = True , 
                                                             color_mode = "rgb",
                                                             class_mode = "categorical",
                                                             subset = "validation",
                                                             seed = 12
                                                            )

    test_generator = test_datagen.flow_from_directory(directory = test_dir,
                                                       target_size = (IMG_HEIGHT ,IMG_WIDTH),
                                                        batch_size = BATCH_SIZE,
                                                        shuffle  = False , 
                                                        color_mode = "rgb",
                                                        class_mode = "categorical",
                                                        seed = 12
                                                      )



    def feature_extractor(inputs):
        feature_extractor = tf.keras.applications.DenseNet169(input_shape=(IMG_HEIGHT,IMG_WIDTH, 3),
                                                   include_top=False,
                                                   weights="imagenet")(inputs)
        
        return feature_extractor

    def classifier(inputs):
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        x = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer = tf.keras.regularizers.l2(0.01))(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(1024, activation="relu", kernel_regularizer = tf.keras.regularizers.l2(0.01))(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(512, activation="relu", kernel_regularizer = tf.keras.regularizers.l2(0.01))(x)
        x = tf.keras.layers.Dropout(0.5) (x)
        x = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="classification")(x)
        
        return x

    def final_model(inputs):
        densenet_feature_extractor = feature_extractor(inputs)
        classification_output = classifier(densenet_feature_extractor)
        
        return classification_output

    def define_compile_model():
        
        inputs = tf.keras.layers.Input(shape=(IMG_HEIGHT ,IMG_WIDTH,3))
        classification_output = final_model(inputs) 
        model = tf.keras.Model(inputs=inputs, outputs = classification_output)
         
        model.compile(optimizer=tf.keras.optimizers.SGD(0.1), 
                    loss='categorical_crossentropy',
                    metrics = ['accuracy'])
      
        return model




    ### HYPERPARAM SEARCH
    LEARNING_RATES = [0.001]

    for learning_rate in LEARNING_RATES:
        LR = learning_rate
        
        print("CURRENTLY TESTING: BATCH SIZE {} and LEARNING RATE {}".format(BATCH_SIZE, LR))
        print()

        model = define_compile_model()
        clear_output()

        # Freezing the feature extraction layers
        model.layers[1].trainable = False

        model.summary()


        earlyStoppingCallback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                                 patience=EARLY_STOPPING_CRITERIA,
                                                                 verbose= 1 ,
                                                                 restore_best_weights=True
                                                                )

        history = model.fit(x = train_generator,
                            epochs = EPOCHS ,
                            validation_data = validation_generator , 
                            callbacks= [earlyStoppingCallback])

        history = pd.DataFrame(history.history)


        # Un-Freezing the feature extraction layers for fine tuning
        model.layers[1].trainable = True

        model.compile(optimizer=tf.keras.optimizers.SGD(0.001), #lower learning rate
                        loss='categorical_crossentropy',
                        metrics = ['accuracy'])

        history_ = model.fit(x = train_generator,epochs = FINE_TUNING_EPOCHS ,validation_data = validation_generator)
        history = history.append(pd.DataFrame(history_.history) , ignore_index=True)


        model.evaluate(test_generator)
        preds = model.predict(test_generator)
        y_preds = np.argmax(preds , axis = 1 )
        y_test = np.array(test_generator.labels)


    ## CONVERSION TO DENSENET

    #mlmodel = ct.convert(model)
    #mlmodel.save("densenetmodel")

def saliency_map(img_name):
    _img = keras.preprocessing.image.load_img(img_name,target_size=(48,48))
    img = keras.preprocessing.image.img_to_array(_img)
    img = img.reshape((1, *img.shape))
    pred = model.predict(img)

    images = tf.Variable(img, dtype=float)

    with tf.GradientTape() as tape:
        pred = model(images, training=False)
        class_idxs_sorted = np.argsort(pred.numpy().flatten())[::-1]
        loss = pred[0][class_idxs_sorted[0]]

    grads = tape.gradient(loss, images)

    dgrad_abs = tf.math.abs(grads)
    dgrad_max_ = np.max(dgrad_abs, axis=3)[0]

    ## normalize to range between 0 and 1
    arr_min, arr_max  = np.min(dgrad_max_), np.max(dgrad_max_)
    grad_eval = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-18)

    new_name = "saliency_" + img_name
    plt.imsave(new_name, grad_eval, cmap="jet")

saliency_map("happy.png")
saliency_map("angry.png")
saliency_map("disgusted.png")
saliency_map("fearful.png")
saliency_map("neutral.png")
saliency_map("sad.png")
saliency_map("surprised.png")
