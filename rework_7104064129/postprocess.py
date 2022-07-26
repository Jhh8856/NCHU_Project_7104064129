import binary_loss
import Otsu

import numpy as np

#should rewrite by argparse
dataset = "ph2"
bits = binary_loss.bits

X_train_reg = np.load('{}_X_train_reg.npy'.format(dataset))
y_train_reg = np.load('{}_y_train_reg.npy'.format(dataset))
result_train = np.load('{}_classify_result_train.npy'.format(dataset))

X_test_reg = np.load('{}_X_test_reg.npy'.format(dataset))
y_test_reg = np.load('{}_y_test_reg.npy'.format(dataset))
result_test = np.load('{}_classify_result_test.npy'.format(dataset))

X_val_reg = np.load('{}_X_val_reg.npy'.format(dataset))
y_val_reg = np.load('{}_y_val_reg.npy'.format(dataset))
result_val = np.load('{}_classify_result_val.npy'.format(dataset))

def group(result):
    # counter = 0
    group = []
    for data in result:
        # counter += 1
        # print("processing:", "group", counter)
        if data[0] >= np.max(data):
            group.append(1)
        elif data[1] >= np.max(data):
            group.append(2)
        elif data[2] >= np.max(data):
            group.append(3)
        elif data[3] >= np.max(data):
            group.append(4)
    
    return group

def split(X, y, result, normalize=False):
    im1 = []
    im2 = []
    im3 = []
    im4 = []
    dmos1 = []
    dmos2 = []
    dmos3 = []
    dmos4 = []
    g = group(result)
    ynor = Otsu.minmax(y)
    ynor = ynor.flatten()
    print(ynor.shape)
    if normalize == False:
        if len(X) == len(g):
            for i in range(len(g)):
                # print("processing:", "dmos", i)
                if g[i] == 1:
                    im1.append(X[i])
                    dmos1.append(y[i])
                elif g[i] == 2:
                    im2.append(X[i])
                    dmos2.append(y[i])     
                elif g[i] == 3:
                    im3.append(X[i])
                    dmos3.append(y[i])
                elif g[i] == 4:
                    im4.append(X[i])
                    dmos4.append(y[i])
    elif normalize == True:
        if len(X) == len(g):
            for i in range(len(g)):
                # print("processing:", "dmos", i)
                if g[i] == 1:
                    im1.append(X[i])
                    dmos1.append(ynor[i])
                elif g[i] == 2:
                    im2.append(X[i])
                    dmos2.append(ynor[i])
                elif g[i] == 3:
                    im3.append(X[i])
                    dmos3.append(ynor[i])
                elif g[i] == 4:
                    im4.append(X[i])
                    dmos4.append(ynor[i])

    return [[np.array(im1),
             np.array(im2), 
             np.array(im3), 
             np.array(im4)], 
            [dmos1,
             dmos2,
             dmos3, 
             dmos4]]

im_train, im_train_dmos = split(X_train_reg, y_train_reg, result_train)
im_test, im_test_dmos = split(X_test_reg, y_test_reg, result_test)
im_val, im_val_dmos = split(X_val_reg, y_val_reg, result_val)

#for MSE
im_mse_train, im_mse_train_dmos = split(X_train_reg, y_train_reg, result_train, normalize=True)
im_mse_test, im_mse_test_dmos = split(X_test_reg, y_test_reg, result_test, normalize=True)
im_mse_val, im_mse_val_dmos = split(X_val_reg, y_val_reg, result_val, normalize=True)

for i in range(bits):
    # print(15*i)
    np.save('{}_im{}_X_train.npy'.format(dataset, i+1), im_train[i])
    np.save('{}_im{}_dmos_train.npy'.format(dataset, i+1), im_train_dmos[i])
    np.save('{}_im{}_y_train.npy'.format(dataset, i+1), [ binary_loss.de2bin(im_train_dmos[i][j]-15*i) for j in range(len(im_train_dmos[i])) ])
    
    np.save('{}_im{}_X_test.npy'.format(dataset, i+1), im_test[i])
    np.save('{}_im{}_dmos_test.npy'.format(dataset, i+1), im_test_dmos[i])
    np.save('{}_im{}_y_test.npy'.format(dataset, i+1), [ binary_loss.de2bin(im_test_dmos[i][j]-15*i) for j in range(len(im_test_dmos[i])) ])
    
    np.save('{}_im{}_X_val.npy'.format(dataset, i+1), im_val[i])
    np.save('{}_im{}_dmos_val.npy'.format(dataset, i+1), im_val_dmos[i])
    np.save('{}_im{}_y_val.npy'.format(dataset, i+1), [ binary_loss.de2bin(im_val_dmos[i][j]-15*i) for j in range(len(im_val_dmos[i])) ])
    
    # np.save('train_min.npy', train_min)
    # np.save('test_min.npy', test_min)
    # np.save('val_min.npy', val_min)
    
    #for MSE
    np.save('{}_mse_im{}_X_train.npy'.format(dataset, i+1), im_mse_train[i])
    np.save('{}_mse_im{}_dmos_train.npy'.format(dataset, i+1), im_mse_train_dmos[i])
    
    np.save('{}_mse_im{}_X_test.npy'.format(dataset, i+1), im_mse_test[i])
    np.save('{}_mse_im{}_dmos_test.npy'.format(dataset, i+1), im_mse_test_dmos[i])
    
    np.save('{}_mse_im{}_X_val.npy'.format(dataset, i+1), im_mse_val[i])
    np.save('{}_mse_im{}_dmos_val.npy'.format(dataset, i+1), im_mse_val_dmos[i])
