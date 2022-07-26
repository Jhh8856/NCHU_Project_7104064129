import Otsu

import os
import numpy as np
import scipy.io as sio

# from PIL import Image
import cv2

#should rewrite by argparse
data_path = './LIVE3DIQD_origin/Phase1/3d_IQA_database/'
data = sio.loadmat(str(data_path+'data.mat'))
lst_dir = []
#blur
#ff
#jp2k
#jpeg
#wn
for array in data['img_names'][0]:
    #if 'jpeg' in array[0]:
    lst_dir.append(array[0])
dmos = data['dmos']

mini = min(dmos)[0]
# print(mini)
np.save("mini.npy", mini)

dmos = np.array([dmos[i] + abs(mini) for i in range(len(dmos))])

patch_size = 224
patch_count = 12
seed = 42

def load_img(patch_size=64, patch_count=60, data_path=data_path, lst_dir=lst_dir, to_reg=False):
    lmap = []
    rmap = []
    dmap = []
    train_data = []
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    
    count = 0
    for names in lst_dir:
        
        lnames = names.split('.')[0] + '_l.' + names.split('.')[1]
        limage = cv2.imread(os.path.join(data_path, lnames))
        lgrayscale = cv2.cvtColor(limage, cv2.COLOR_BGR2GRAY)
        larr = np.array(lgrayscale)
        
        temp0 = [np.random.randint(0, larr.shape[0]-patch_size) for i in range(patch_count)]
        temp1 = [np.random.randint(0, larr.shape[1]-patch_size) for i in range(patch_count)]
        
        for i in range(patch_count):
            lmap.append(larr[temp0[i]:temp0[i]+patch_size, temp1[i]:temp1[i]+patch_size])
            
        rnames = names.split('.')[0] + '_r.' + names.split('.')[1]
        rimage = cv2.imread(os.path.join(data_path, rnames))
        rgrayscale = cv2.cvtColor(rimage, cv2.COLOR_BGR2GRAY)
        rarr = np.array(rgrayscale)
        
        for i in range(patch_count):
            rmap.append(rarr[temp0[i]:temp0[i]+patch_size, temp1[i]:temp1[i]+patch_size])

        disparity = stereo.compute(lgrayscale, rgrayscale)
        darr = np.array(disparity)
        for i in range(patch_count):
            # disparity = np.array(stereo.compute(lmap[i],rmap[i]))
            dmap.append(darr[temp0[i]:temp0[i]+patch_size, temp1[i]:temp1[i]+patch_size])
            # if np.shape(disparity) != np.shape(lmap[i]):
            #     print(np.shape(lmap[i]))
            #     print(np.shape(rmap[i]))
            #     print(np.shape(disparity))
            #     raise ValueError
        count += 1
        print('patch process:patch {}...'.format(count))
    
    if to_reg is True:
        for i in range(len(lmap)):
            train_data.append([Otsu.normalize(lmap[i]),
                               Otsu.otsu(lmap[i]),
                               Otsu.otsu_b(dmap[i]),
                               Otsu.otsu(dmap[i]),
                               Otsu.otsu_f(dmap[i])])
            train_data.append([Otsu.normalize(rmap[i]),
                               Otsu.otsu(rmap[i]),
                               Otsu.otsu_b(dmap[i]),
                               Otsu.otsu(dmap[i]),
                               Otsu.otsu_f(dmap[i])])

            print('append(bin):patch {}...'.format(i+1))
        return np.moveaxis(np.array(train_data), 1, 3) #channel_last
    else:
        for i in range(len(lmap)):
            train_data.append([Otsu.normalize(lmap[i]),
                               Otsu.otsu(lmap[i]),
                               Otsu.otsu(dmap[i])])
            train_data.append([Otsu.normalize(rmap[i]),
                               Otsu.otsu(rmap[i]),
                               Otsu.otsu(dmap[i])])

            print('append:patch {}...'.format(i+1))
        return np.moveaxis(np.array(train_data), 1, 3) #channel_last

def y_patch(dmos, patch_count=60):
    lst = []
    for score in dmos:
            for i in range(patch_count):
                for j in range(2):#left & right
                    lst.append(score[0])
    
    return np.array(lst)

def group(dmos):
    group = []
    for score in dmos:
        if score < 15:
            group.append(0)
        elif 15 <= score < 30:
            group.append(1)
        elif 30 <= score < 45:
            group.append(2)
        elif score >= 45:
            group.append(3)

    return np.array(group)

dataset = load_img(patch_size=patch_size, patch_count=patch_count)
dataset_bin = load_img(patch_size=patch_size, patch_count=patch_count, to_reg=True)
dmos = y_patch(dmos, patch_count=patch_count).reshape(-1, 1)
group = group(dmos).flatten().reshape(-1, 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset, group, random_state=seed, shuffle=True, train_size=0.8) #spilt dataset
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=seed, shuffle=True, train_size=0.875) #0.8*0.875=0.7

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(dataset_bin, dmos, shuffle=True, random_state=seed, train_size=0.8)
X_train_reg, X_val_reg, y_train_reg, y_val_reg = train_test_split(X_train_reg, y_train_reg, shuffle=True, random_state=seed, train_size=0.875)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train) # one-hot encoding
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

np.save('ph1_X_train', X_train)
np.save('ph1_y_train', y_train)
np.save('ph1_X_test', X_test)
np.save('ph1_y_test', y_test)
np.save('ph1_X_val', X_val)
np.save('ph1_y_val', y_val)

np.save('ph1_X_train_reg', X_train_reg)
np.save('ph1_y_train_reg', y_train_reg)
np.save('ph1_X_test_reg', X_test_reg)
np.save('ph1_y_test_reg', y_test_reg)
np.save('ph1_X_val_reg', X_val_reg)
np.save('ph1_y_val_reg', y_val_reg)