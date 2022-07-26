import tensorflow as tf
# tf.config.run_functions_eagerly(True)

import tensorflow.experimental.numpy as tnp
tnp.experimental_enable_numpy_behavior()

import tensorflow.keras.backend as K

import numpy as np

#should rewrite by argparse
bits = 4
mini = np.load('mini.npy')
mini = abs(mini)
minimum = tf.constant(mini, dtype=tf.float32)
w = [128, 64, 32, 16, 8, 4, 2, 1][8-bits:]
weight = tf.constant(w, dtype=tf.float32)

def roundup(x):
    return tf.round(x)

def tee(x, k=20):
    temp = tf.subtract(x, 0.5)
    tempk = tf.multiply(temp, k)
    return tf.multiply(0.5, tf.add(1.0, tf.math.tanh(tempk)))
    
# r = tf.constant([0.6, 0.4, 0.48, 0.5])
# print(tee(r))

def de2bin(x):
    x = int(x)
    lst = []
    for i in range(bits):
        if x >= 0: 
            if x >= 2**(bits-i-1):
                lst.append(1)
                x = x - 2**(bits-i-1)
            else:
                lst.append(0)
        else:
            lst.append(0)
            
    return roundup(tf.convert_to_tensor(np.array(lst), dtype=tf.int32))

# print(de2bin(9.74646165631))

def weighted(x):
    value = tf.multiply(x, weight)
    return value

def bin2de(x):
    value = tf.multiply(x, weight)
    value = tf.reduce_sum(value)
    return value.numpy()

# print(bin2de([1, 0, 0, 1]))
def binloss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    # loss = tf.math.abs(weighted(tf.subtract(y_true, y_pred)))
    loss = tf.math.abs(tf.subtract(y_true, y_pred))
    return tf.reduce_sum(loss, axis=-1)

def binloss_weighted(y_true, ypred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    loss = tf.math.abs(weighted(tf.subtract(y_true, y_pred)))
    # loss = tf.math.abs(tf.subtract(y_true, y_pred))
    return tf.reduce_sum(loss, axis=-1)

# print(binloss([0, 0, 0, 1], [1, 0, 1, 0]))

#def bin_loss(y_ture, y_pred):
    # ytruth and ypred are both 4bits
    # using xor
    # does ypred list-form?
    # try:
        #a = (np.logical_xor(y_ture, y_pred))
        #print('np_xor長這樣 ', a)
        #b = np.logical_xor(y_ture, y_pred) * 1
        #print('np_xor * 1長這樣 ', b)
        #c = list((np.logical_xor(y_ture, y_pred)) * 1)
        #print('list np_xor * 1長這樣 ', c)
        
    #     return list((np.logical_xor(y_ture, y_pred)) * 1)
    # except ValueError : #list長度不一樣
    #     print('len y_ture : {} , are not same as \nlen y_pred : {}'.format(len(y_ture), len(y_pred)))
        
# def binloss(y_true, y_pred):
#     y_true = tf.cast(y_true, tf.bool)
#     y_pred = tf.cast(y_pred, tf.bool)
#     value = tf.cast(tf.math.logical_xor(y_true, y_pred), tf.int32)
#     # print(value)
#     weight = tf.constant([8, 4, 2, 1])
#     # print(weight)
#     loss = tf.multiply(value, weight)
#     # print(loss)
#     loss = tf.math.reduce_sum(loss)
#     # print(loss)
#     return loss

# r1 = tf.convert_to_tensor([1, 1, 0, 0], dtype=tf.float32)
# r2 = tf.convert_to_tensor([0, 1, 0, 1], dtype=tf.float32)
# print(binloss(r1, r2))

# @tf.function
# def tee(x):
    # temp = tf.identity(x)
    
    # a = tf.constant(1)
    # b = tf.constant(0)
    # c = tf.constant(0.5)

    # f1 = lambda:a
    # f2 = lambda:b
    
    # r = tf.cond(tf.math.greater_equal(temp, c), f1, f2)
    
    # r = tf.round(x)
    
    # return r
    
# def PLCC(y_true, y_pred):
    
#     x = y_true
#     y = y_pred
#     mx = K.mean(x, axis=0)
#     my = K.mean(y, axis=0)
#     xm, ym = x - mx, y - my
#     r_num = K.sum(xm * ym)
#     x_square_sum = K.sum(xm * xm)
#     y_square_sum = K.sum(ym * ym)
#     r_den = K.sqrt(x_square_sum * y_square_sum)
#     r = r_num / r_den

#     return K.mean(r)

#https://www.kaggle.com/carlolepelaars/understanding-the-metric-spearman-s-rho
# def rank(arr):
#     temp = arr.argsort(kind="stable")
#     ranks = tnp.empty_like(temp)
#     ranks[temp] = tnp.arange(tnp.shape(arr)[0])
#     return ranks

# def SROCC(y_true, y_pred):

#     true_rank = rank(y_true)
#     pred_rank = rank(y_pred)
    
#     return PLCC(true_rank, pred_rank)[1][0]