
import tensorflow as tf
from tensorflow import keras
# tf.debugging.set_log_device_placement(True)

import numpy as np

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

dataset = "ph1"

#should rewrite by argparse
X_train = np.load('{}_X_train.npy'.format(dataset))
y_train = np.load('{}_y_train.npy'.format(dataset))
X_test = np.load('{}_X_test.npy'.format(dataset))
y_test = np.load('{}_y_test.npy'.format(dataset))
X_val = np.load('{}_X_val.npy'.format(dataset))
y_val = np.load('{}_y_val.npy'.format(dataset))

input_shape=(224, 224, 3)
patch_count = 8
epochs=200
batch_size=200
learning_rate=1e-3
decay=0
seed=54321

#main
# https://github.com/qubvel/efficientnet
import efficientnet
from efficientnet.efficientnet import tfkeras as efn
firstsmx = efn.EfficientNetB0(weights=None, classes=4)

firstsmx.summary()

firstsmx.compile(optimizer=keras.optimizers.SGD(lr=learning_rate, decay=decay),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

fitting = firstsmx.fit(X_train, y_train,
                    epochs=epochs, batch_size=batch_size,
                    validation_data=(X_val, y_val))

evl = firstsmx.evaluate(X_test, y_test)
print('test_loss:', evl[0], 'test_accuracy:', evl[1])

result1 = firstsmx.predict(X_train)
result2 = firstsmx.predict(X_test)
result3 = firstsmx.predict(X_val)

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

#save
plt.savefig('{}x{}_{}patches.pdf'.format(input_shape[0], input_shape[1], patch_count))
firstsmx.save('{}_classify.hdf5'.format(dataset))
np.save('{}_classify_result_train'.format(dataset), result1)
np.save('{}_classify_result_test'.format(dataset), result2)
np.save('{}_classify_result_val'.format(dataset), result3)
np.save('{}_classify_eval'.format(dataset), evl)