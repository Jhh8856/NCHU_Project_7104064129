import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import utils
# tf.debugging.set_log_device_placement(True)

import numpy as np
# from . import imagenet_utils


_KERAS_BACKEND = None
_KERAS_LAYERS = None
_KERAS_MODELS = None
_KERAS_UTILS = None


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
epochs=100
batch_size=30
learning_rate=1e-3
decay=0
seed=12345

#main
#https://github.com/keras-team/keras-applications/blob/master/keras_applications/densenet.py
def _obtain_input_shape(input_shape,
                        default_size,
                        min_size,
                        data_format,
                        require_flatten,
                        weights=None):
    """Internal utility to compute/validate a model's input shape.
    # Arguments
        input_shape: Either None (will return the default network input shape),
            or a user-provided shape to be validated.
        default_size: Default input width/height for the model.
        min_size: Minimum input width/height accepted by the model.
        data_format: Image data format to use.
        require_flatten: Whether the model is expected to
            be linked to a classifier via a Flatten layer.
        weights: One of `None` (random initialization)
            or 'imagenet' (pre-training on ImageNet).
            If weights='imagenet' input channels must be equal to 3.
    # Returns
        An integer shape tuple (may include None entries).
    # Raises
        ValueError: In case of invalid argument values.
    """
    if weights != 'imagenet' and input_shape and len(input_shape) == 3:
        if data_format == 'channels_first':
            if input_shape[0] not in {1, 3}:
                warnings.warn(
                    'This model usually expects 1 or 3 input channels. '
                    'However, it was passed an input_shape with ' +
                    str(input_shape[0]) + ' input channels.')
            default_shape = (input_shape[0], default_size, default_size)
        else:
            if input_shape[-1] not in {1, 3}:
                warnings.warn(
                    'This model usually expects 1 or 3 input channels. '
                    'However, it was passed an input_shape with ' +
                    str(input_shape[-1]) + ' input channels.')
            default_shape = (default_size, default_size, input_shape[-1])
    else:
        if data_format == 'channels_first':
            default_shape = (3, default_size, default_size)
        else:
            default_shape = (default_size, default_size, 3)
    if weights == 'imagenet' and require_flatten:
        if input_shape is not None:
            if input_shape != default_shape:
                raise ValueError('When setting `include_top=True` '
                                 'and loading `imagenet` weights, '
                                 '`input_shape` should be ' +
                                 str(default_shape) + '.')
        return default_shape
    if input_shape:
        if data_format == 'channels_first':
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError(
                        '`input_shape` must be a tuple of three integers.')
                if input_shape[0] != 3 and weights == 'imagenet':
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape=' + str(input_shape) + '`')
                if ((input_shape[1] is not None and input_shape[1] < min_size) or
                   (input_shape[2] is not None and input_shape[2] < min_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_size) + 'x' + str(min_size) +
                                     '; got `input_shape=' +
                                     str(input_shape) + '`')
        else:
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError(
                        '`input_shape` must be a tuple of three integers.')
                if input_shape[-1] != 3 and weights == 'imagenet':
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape=' + str(input_shape) + '`')
                if ((input_shape[0] is not None and input_shape[0] < min_size) or
                   (input_shape[1] is not None and input_shape[1] < min_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_size) + 'x' + str(min_size) +
                                     '; got `input_shape=' +
                                     str(input_shape) + '`')
    else:
        if require_flatten:
            input_shape = default_shape
        else:
            if data_format == 'channels_first':
                input_shape = (3, None, None)
            else:
                input_shape = (None, None, 3)
    if require_flatten:
        if None in input_shape:
            raise ValueError('If `include_top` is True, '
                             'you should specify a static `input_shape`. '
                             'Got `input_shape=' + str(input_shape) + '`')
    return input_shape

def dense_block(x, blocks, name):
    """A dense block.
    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction, name):
    """A transition block.
    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv2D(int(backend.int_shape(x)[bn_axis] * reduction), 1,
                      use_bias=False,
                      name=name + '_conv')(x)
    x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def conv_block(x, growth_rate, name):
    """A building block for a dense block.
    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.
    # Returns
        Output tensor for the block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    x1 = layers.BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + '_0_bn')(x)
    x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
    x1 = layers.Conv2D(4 * growth_rate, 1,
                       use_bias=False,
                       name=name + '_1_conv')(x1)
    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   name=name + '_1_bn')(x1)
    x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = layers.Conv2D(growth_rate, 3,
                       padding='same',
                       use_bias=False,
                       name=name + '_2_conv')(x1)
    x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def DenseNet(blocks,
             include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             **kwargs):
    """Instantiates the DenseNet architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        blocks: numbers of building blocks for the four dense layers.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `'channels_last'` data format)
            or `(3, 224, 224)` (with `'channels_first'` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: optional pooling mode for feature extraction
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
    # global backend, layers, models, keras_utils
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
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
    x = layers.Activation('relu', name='conv1/relu')(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')

    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
    x = layers.Activation('relu', name='relu')(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        # inputs = keras_utils.get_source_inputs(input_tensor)
        inputs = keras.utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    if blocks == [6, 12, 24, 16]:
        model = models.Model(inputs, x, name='densenet121')
    elif blocks == [6, 12, 32, 32]:
        model = models.Model(inputs, x, name='densenet169')
    elif blocks == [6, 12, 48, 32]:
        model = models.Model(inputs, x, name='densenet201')
    else:
        model = models.Model(inputs, x, name='densenet')

    # Load weights.
    # if weights == 'imagenet':
    #     if include_top:
    #         if blocks == [6, 12, 24, 16]:
    #             weights_path = keras_utils.get_file(
    #                 'densenet121_weights_tf_dim_ordering_tf_kernels.h5',
    #                 DENSENET121_WEIGHT_PATH,
    #                 cache_subdir='models',
    #                 file_hash='9d60b8095a5708f2dcce2bca79d332c7')
    #         elif blocks == [6, 12, 32, 32]:
    #             weights_path = keras_utils.get_file(
    #                 'densenet169_weights_tf_dim_ordering_tf_kernels.h5',
    #                 DENSENET169_WEIGHT_PATH,
    #                 cache_subdir='models',
    #                 file_hash='d699b8f76981ab1b30698df4c175e90b')
    #         elif blocks == [6, 12, 48, 32]:
    #             weights_path = keras_utils.get_file(
    #                 'densenet201_weights_tf_dim_ordering_tf_kernels.h5',
    #                 DENSENET201_WEIGHT_PATH,
    #                 cache_subdir='models',
    #                 file_hash='1ceb130c1ea1b78c3bf6114dbdfd8807')
    #     else:
    #         if blocks == [6, 12, 24, 16]:
    #             weights_path = keras_utils.get_file(
    #                 'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5',
    #                 DENSENET121_WEIGHT_PATH_NO_TOP,
    #                 cache_subdir='models',
    #                 file_hash='30ee3e1110167f948a6b9946edeeb738')
    #         elif blocks == [6, 12, 32, 32]:
    #             weights_path = keras_utils.get_file(
    #                 'densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5',
    #                 DENSENET169_WEIGHT_PATH_NO_TOP,
    #                 cache_subdir='models',
    #                 file_hash='b8c4d4c20dd625c148057b9ff1c1176b')
    #         elif blocks == [6, 12, 48, 32]:
    #             weights_path = keras_utils.get_file(
    #                 'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5',
    #                 DENSENET201_WEIGHT_PATH_NO_TOP,
    #                 cache_subdir='models',
    #                 file_hash='c13680b51ded0fb44dff2d8f86ac8bb1')
    #     model.load_weights(weights_path)
    # elif weights is not None:
    #     model.load_weights(weights)

    return model


def DenseNet121(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                **kwargs):
    return DenseNet([6, 12, 24, 16],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes,
                    **kwargs)


def DenseNet169(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                **kwargs):
    return DenseNet([6, 12, 32, 32],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes,
                    **kwargs)


def DenseNet201(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                **kwargs):
    return DenseNet([6, 12, 48, 32],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes,
                    **kwargs)

firstsmx = DenseNet121(include_top=True,
                weights=None,
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=4)

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
firstsmx.save('classify.hdf5')
np.save('classify_result_train_DanseNet', result1)
np.save('classify_result_test_DanseNet', result2)
np.save('classify_result_val_DanseNet', result3)
np.save('classify_eval_NASNet', evl)