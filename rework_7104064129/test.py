# import tensorflow as tf
# gpu = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpu[0], True)
# print(tf.__version__)

import numpy as np
images = np.full((256, 256, 3), 128)
for i in range(128):
    images[i+64] = 255

# images = images.reshape(1, 256, 256, 3)