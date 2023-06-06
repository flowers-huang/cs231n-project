import coremltools as ct
import tensorflow as tf

model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax),
    ]
)


(X_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# fit the model
model.fit(X_train, y_train, batch_size=32)

mlmodel = ct.convert(model)

mlmodel.save('test_model')

# for alexnet conversion

#mlmodel = ct.convert('AlexNet_saved_model', source="tensorflow",convert_to="mlprogram")
#mlmodel.save('mlModel')


