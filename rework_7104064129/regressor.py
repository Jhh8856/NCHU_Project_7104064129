import binary_loss

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
# from keras.utils.generic_utils import get_custom_objects
# tf.debugging.set_log_device_placement(True)

import numpy as np

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

dataset = "ph2"
group = 1

#should rewrite by argparse
X_train = np.load('{}_im{}_X_train.npy'.format(dataset, group))
y_train = np.load('{}_im{}_y_train.npy'.format(dataset, group))
X_test = np.load('{}_im{}_X_test.npy'.format(dataset, group))
y_test = np.load('{}_im{}_y_test.npy'.format(dataset, group))
X_val = np.load('{}_im{}_X_val.npy'.format(dataset, group))
y_val = np.load('{}_im{}_y_val.npy'.format(dataset, group))

input_shape = (224, 224, 5)
epochs = 400
batch_size = 200
learning_rate = 1e-5
decay = 0
seed = 42
bits = binary_loss.bits

#main
# https://github.com/qubvel/efficientnet
regress = keras.models.Sequential([
    
    keras.layers.Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same', kernel_initializer=keras.initializers.RandomNormal(seed=seed), input_shape=input_shape),
    keras.layers.Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same'),
    keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
    
    keras.layers.Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='same', kernel_initializer=keras.initializers.RandomNormal(seed=seed)),
    keras.layers.Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='same'),
    keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
    
    keras.layers.Conv2D(256, (3, 3), activation='relu', strides=(1, 1), padding='same', kernel_initializer=keras.initializers.RandomNormal(seed=seed)),
    keras.layers.Conv2D(256, (3, 3), activation='relu', strides=(1, 1), padding='same'),
    keras.layers.Conv2D(256, (3, 3), activation='relu', strides=(1, 1), padding='same'),
    keras.layers.Conv2D(256, (3, 3), activation='relu', strides=(1, 1), padding='same'),
    keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
    
    keras.layers.Conv2D(512, (3, 3), activation='relu', strides=(1, 1), padding='same', kernel_initializer=keras.initializers.RandomNormal(seed=seed)),
    keras.layers.Conv2D(512, (3, 3), activation='relu', strides=(1, 1), padding='same'),
    keras.layers.Conv2D(512, (3, 3), activation='relu', strides=(1, 1), padding='same'),
    keras.layers.Conv2D(512, (3, 3), activation='relu', strides=(1, 1), padding='same'),
    keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
    
    keras.layers.Conv2D(512, (3, 3), activation='relu', strides=(1, 1), padding='same', kernel_initializer=keras.initializers.RandomNormal(seed=seed)),
    keras.layers.Conv2D(512, (3, 3), activation='relu', strides=(1, 1), padding='same'),
    keras.layers.Conv2D(512, (3, 3), activation='relu', strides=(1, 1), padding='same'),
    keras.layers.Conv2D(512, (3, 3), activation='relu', strides=(1, 1), padding='same'),
    keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
    
    keras.layers.Dropout(0.1),
    
    keras.layers.Flatten(),
    
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dense(binary_loss.bits, activation='sigmoid'),
    keras.layers.Lambda(binary_loss.tee)
    ])

regress.summary()

# CC = binary_loss.PLCC
# bce = tf.keras.losses.BinaryCrossentropy()
#binloss = binary_loss.binloss
wbinloss = binary_loss.binloss
rmse = tf.keras.metrics.RootMeanSquaredError()

regress.compile(optimizer=keras.optimizers.SGD(lr=learning_rate, decay=decay),
              loss=wbinloss,
              metrics = ['MAE', 'MSE', rmse])

fitting = regress.fit(X_train, y_train,
                    epochs=epochs, batch_size=batch_size,
                    validation_data=(X_val, y_val))

evl = regress.evaluate(X_test, y_test)
print('test_loss:', evl)

result1 = regress.predict(X_train)
result2 = regress.predict(X_test)
result3 = regress.predict(X_val)

#visualization
loss = fitting.history['loss']
val_loss = fitting.history['val_loss']

plt.figure(figsize=(20,10))
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Training and Validation loss')

ax1.plot(range(1, epochs+1), loss, 'r', label='Training loss')
ax1.legend()

ax2.plot(range(1, epochs+1), val_loss, 'b', label='validation loss')
ax2.legend()

np.save('{}_im{}_regressor_result_train'.format(dataset, group), result1)
np.save('{}_im{}_regressor_result_test'.format(dataset, group), result2)
np.save('{}_im{}_regressor_result_val'.format(dataset, group), result3)
np.save('{}_im{}_regressor_eval'.format(dataset, group), evl)
