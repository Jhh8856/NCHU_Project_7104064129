import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow import keras
# tf.debugging.set_log_device_placement(True)
from keras import backend as K
import numpy as np

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

dataset = "mcl"

#should rewrite by argparse
X_train = np.load('{}_X_train.npy'.format(dataset))
y_train = np.load('{}_y_train.npy'.format(dataset))
X_test = np.load('{}_X_test.npy'.format(dataset))
y_test = np.load('{}_y_test.npy'.format(dataset))
X_val = np.load('{}_X_val.npy'.format(dataset))
y_val = np.load('{}_y_val.npy'.format(dataset))

input_shape=(224, 224, 3)
patch_count = 10
epochs=400
batch_size = 50  # 調
learning_rate = 1e-3
decay=0
seed=54321

#main
firstsmx = keras.models.Sequential([
       
    keras.layers.Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same', kernel_initializer=keras.initializers.RandomNormal(seed=seed), input_shape=input_shape),
    keras.layers.Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same'),
    # keras.layers.Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same'),
    # keras.layers.Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same'),
    keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
    
    keras.layers.Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='same', kernel_initializer=keras.initializers.RandomNormal(seed=seed)),
    keras.layers.Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='same'),
    # keras.layers.Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='same'),
    # keras.layers.Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='same'),
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
    
    # keras.layers.Conv2D(512, (3, 3), activation='relu', strides=(1, 1), padding='same', kernel_initializer=keras.initializers.RandomNormal(seed=seed)),
    keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
    
    keras.layers.Dropout(0.1),
    
    keras.layers.Flatten(),

    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dense(4, activation='softmax')
    
    ])

firstsmx.summary()

firstsmx.compile(optimizer=keras.optimizers.SGD(lr=learning_rate, decay=decay),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

fitting = firstsmx.fit(X_train, y_train,
                    epochs=epochs, batch_size=batch_size,
                    validation_data=(X_val, y_val))

evl = firstsmx.evaluate(X_test, y_test)
print('test_loss:', evl[0], 'test_accuracy:', evl[1])
np.save('{}_classify_eval'.format(dataset), evl)
K.clear_session()

result1 = firstsmx.predict(X_train)
np.save('{}_classify_result_train'.format(dataset), result1)
K.clear_session()

result2 = firstsmx.predict(X_test)
np.save('{}_classify_result_test'.format(dataset), result2)
K.clear_session()

result3 = firstsmx.predict(X_val)
np.save('{}_classify_result_val'.format(dataset), result3)
K.clear_session()


firstsmx.save('{}_classify.hdf5'.format(dataset))
K.clear_session()

#visualization
loss = fitting.history['loss']
val_loss = fitting.history['val_loss']
accuracy = fitting.history['accuracy']
val_accuracy = fitting.history['val_accuracy']

plt.figure(figsize=(20,10))
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Training and Validation loss & accuracy')

ax1.plot(range(1, epochs+1), loss, 'r', label='Training loss')
ax1.plot(range(1, epochs+1), val_loss, 'b', label='validation loss')
ax1.legend()

ax2.plot(range(1, epochs+1), accuracy, 'r', label='Training accuracy')
ax2.plot(range(1, epochs+1), val_accuracy, 'b', label='validation accuracy')
ax2.legend()

plt.savefig('{}x{}_{}patches.pdf'.format(input_shape[0], input_shape[1], patch_count))