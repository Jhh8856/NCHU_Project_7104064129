import tensorflow as tf
from tensorflow import keras
# tf.debugging.set_log_device_placement(True)
from tensorflow.keras.optimizers import SGD
# tf.debugging.set_log_device_placement(True)

import numpy as np

import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

#should rewrite by argparse
X_train = np.load('ph1_X_train.npy')
y_train = np.load('ph1_y_train.npy')
X_test = np.load('ph1_X_test.npy')
y_test = np.load('ph1_y_test.npy')
X_val = np.load('ph1_X_val.npy')
y_val = np.load('ph1_y_val.npy')

input_shape=(224, 224, 3)
patch_count = 12
epochs=400
batch_size=200
learning_rate = 5e-4
decay=0
seed=12345

bn_axis=3

#https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py
#https://medium.com/@rossleecooloh/%E7%9B%B4%E8%A7%80%E7%90%86%E8%A7%A3resnet-%E7%B0%A1%E4%BB%8B-%E8%A7%80%E5%BF%B5%E5%8F%8A%E5%AF%A6%E4%BD%9C-python-keras-8d1e2e057de2

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    # if backend.image_data_format() == 'channels_last':
    #     bn_axis = 3
    # else:
    #     bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # 這邊就是圖5上的1x1x64降維操作，假設input x的維度是(n, n, 256), channel last
    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)
    
    # 正常的3x3x64卷積操作，Feature Map長寬仍是n x n
    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)
    
    # 最後升維到256，維度(n,n,256) -> 變成可以和(Indentity)input x相加的維度
    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    
    # 相加後做non-linear轉換
    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    """A block that has a conv layer at shortcut.  --> 做shortcut前有卷積(Projection)
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    # if backend.image_data_format() == 'channels_last':
    #     bn_axis = 3
    # else:
    #     bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # 因為是projection shortcut 所以input的x可能跟output維度不同
    # input維度(n,n,256) -->降維 (n,n,64)
    # 如果Strides有改，則利用Strides來改變Feature Map長寬
    x = layers.Conv2D(filters1, (1, 1), strides=strides,  
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)
    
    # (3,3)的kernel, padding都pad好pad滿，不改變Feature Map尺寸大小
    x = layers.Conv2D(filters2, kernel_size, padding='same', 
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)
    
    # 用1x1 conv升維到假設512
    x = layers.Conv2D(filters3, (1, 1), 
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
  
    # 因input維度是256，這邊就需要做projectr將維度升到512相加
    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
    # F(x) + x(升維後的x)
    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

#main
def ResNet50(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             **kwargs):
    """Instantiates the ResNet50 architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    global backend, layers, models, keras_utils
    # backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    # input_shape = _obtain_input_shape(input_shape,
    #                                   default_size=224,
    #                                   min_size=32,
    #                                   data_format=backend.image_data_format(),
    #                                   require_flatten=include_top,
    #                                   weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)  # 配上Pad剛好會砍半FM尺寸
  
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))  # input Channel大小會跟最後最後residual output尺寸一樣
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    # 256-d to 512-d
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a') # projection shortcut
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    # 512-d to 1024-d
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a') # projection shortcut
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    # 1024-d to 2048-d
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)
        else:
            warnings.warn('The output shape of `ResNet50(include_top=False)` '
                          'has been changed since Keras 2.2.0.')

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name='resnet50')

    # Load weights.
    # if weights == 'imagenet':
    #     if include_top:
    #         weights_path = keras_utils.get_file(
    #             'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
    #             WEIGHTS_PATH,
    #             cache_subdir='models',
    #             md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
    #     else:
    #         weights_path = keras_utils.get_file(
    #             'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
    #             WEIGHTS_PATH_NO_TOP,
    #             cache_subdir='models',
    #             md5_hash='a268eb855778b3df3c7506639542a6af')
    #     model.load_weights(weights_path)
    #     if backend.backend() == 'theano':
    #         keras_utils.convert_all_kernels_in_model(model)
    # elif weights is not None:
    #     model.load_weights(weights)

    return model

# firstsmx = ResNet50(include_top=True, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=4)
firstsmx = keras.applications.ResNet50(include_top=True, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=4)

firstsmx.summary()

firstsmx.compile(optimizer=SGD(lr=learning_rate, decay=decay),
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
plt.savefig('{}x{}_{}patches_resnet50.pdf'.format(input_shape[0], input_shape[1], patch_count))
firstsmx.save('classify_resnet50.hdf5')
np.save('classify_result_train_resnet50', result1)
np.save('classify_result_test_resnet50', result2)
np.save('classify_result_val_resnet50', result3)
np.save('classify_eval_resnet50', evl)