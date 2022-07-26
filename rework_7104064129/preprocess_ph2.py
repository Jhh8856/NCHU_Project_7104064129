import os
import numpy as np
import scipy.io as sio
# from PIL import Image
import cv2
import Otsu


#should rewrite by argparse
data_path = './LIVE3DIQD_origin/Phase2/'
data = sio.loadmat(str(data_path+'3DDmosRelease.mat'))
lst_dir = []
for array in data['StiFilename']:
    lst_dir.append(array[0][0])
dmos = data['Dmos'] -30

patch_size = 224
patch_count = 10
seed = 54321

# print(str(data_path+"Stimuli/"+lst_dir[0]))
# test = cv2.imread(str(data_path+"Stimuli/"+lst_dir[0]))
# testgray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
# testsplit1 = testgray[...,:640]
# testsplit2 = testgray[...,640:]

def load_img(patch_size=64, patch_count=60, data_path=data_path, lst_dir=lst_dir, to_reg=False):
    lmap = []
    rmap = []
    dmap = []
    train_data = []
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    
    count = 0
    for names in lst_dir:
        
        image = cv2.imread(os.path.join(data_path, "Stimuli/", names))
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        limage = grayscale[...,:640]
        larr = np.array(limage)
        
        temp0 = [np.random.randint(0, larr.shape[0]-patch_size) for i in range(patch_count)]
        temp1 = [np.random.randint(0, larr.shape[1]-patch_size) for i in range(patch_count)]
        
        for i in range(patch_count):
            lmap.append(larr[temp0[i]:temp0[i]+patch_size, temp1[i]:temp1[i]+patch_size])
            
        rimage = grayscale[...,640:]
        rarr = np.array(rimage)
        
        for i in range(patch_count):
            rmap.append(rarr[temp0[i]:temp0[i]+patch_size, temp1[i]:temp1[i]+patch_size])

        disparity = stereo.compute(limage, rimage)
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

def patch_index(dmos, patch_count=60):
    lst = []
    for index in range(1, len(dmos)+1):
        for i in range(patch_count):
            for j in range(2):  # left & right
                lst.append(index)
    return np.array(lst)

dataset = load_img(patch_size=patch_size, patch_count=patch_count)
dataset_bin = load_img(patch_size=patch_size, patch_count=patch_count, to_reg=True)
image_patch_index = patch_index(dmos, patch_count=patch_count).reshape(-1, 1)  # 第幾張圖的patch
dmos = y_patch(dmos, patch_count=patch_count).reshape(-1, 1)
group = group(dmos).flatten().reshape(-1, 1)

from sklearn.model_selection import train_test_split
# Classify 0.7 train / 0.1 val / 0.2 test
# X_train, X_test, y_train, y_test = train_test_split(dataset, group, random_state=seed, shuffle=False, train_size=0.8) #spilt dataset
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=seed, shuffle=True, train_size=0.875) #0.8*0.875=0.7

# Classify 0.7 train / 0.1 val / 0.2 test + patch_index
X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(dataset, group, image_patch_index, random_state=seed, shuffle=True, train_size=0.8) # spilt dataset
X_train, X_val, y_train, y_val, z_train, z_val = train_test_split(X_train, y_train, z_train, random_state=seed, shuffle=True, train_size=0.875) # 0.8*0.875=0.7

# Classify 0.8 train / 0.2 test + patch_index
# X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(dataset, group, image_patch_index, random_state=seed, shuffle=True, train_size=0.8) # spilt dataset


# Regress 0.7 train / 0.1 val / 0.2 test
# X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(dataset_bin, dmos, random_state=seed, shuffle=False, train_size=0.8)
# X_train_reg, X_val_reg, y_train_reg, y_val_reg = train_test_split(X_train_reg, y_train_reg, random_state=seed, shuffle=False, train_size=0.875)

# Regress 0.7 train / 0.1 val / 0.2 test + patch_index
X_train_reg, X_test_reg, y_train_reg, y_test_reg, z_train_reg, z_test_reg = train_test_split(dataset_bin, dmos, image_patch_index, random_state=seed, shuffle=True, train_size=0.8)
X_train_reg, X_val_reg, y_train_reg, y_val_reg, z_train_reg, z_val_reg = train_test_split(X_train_reg, y_train_reg, z_train_reg, random_state=seed, shuffle=True, train_size=0.875)

# Regress 0.8 train / 0.2 test + patch_index
# X_train_reg, X_test_reg, y_train_reg, y_test_reg, z_train_reg, z_test_reg = train_test_split(dataset_bin, dmos, image_patch_index, random_state=seed, shuffle=True, train_size=0.8)

#-------------------------------------------------------------------------------#
# X_train, X_test, y_train, y_test = train_test_split(dataset, group, random_state=seed, train_size=0.8) #spilt dataset
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=seed, train_size=0.875) #0.8*0.875=0.7

# X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(dataset_bin, dmos, random_state=seed, train_size=0.8)
# # X_train_reg, X_val_reg, y_train_reg, y_val_reg = train_test_split(X_train_reg, y_train_reg, random_state=seed, train_size=0.875)
#
# X_train, X_test, y_train, y_test = train_test_split(dataset, group, random_state=seed, shuffle=False, train_size=0.8) #spilt dataset
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=seed, shuffle=False, train_size=0.875) #0.8*0.875=0.7
#
# X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(dataset_bin, dmos, random_state=seed, shuffle=False, train_size=0.8)
# X_train_reg, X_val_reg, y_train_reg, y_val_reg = train_test_split(X_train_reg, y_train_reg, random_state=seed, shuffle=False, train_size=0.875)
#-------------------------------------------------------------------------------#
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train) # one-hot encoding
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

np.save('ph2_X_train', X_train)
np.save('ph2_y_train', y_train)
np.save('ph2_z_train', z_train)

np.save('ph2_X_test', X_test)
np.save('ph2_y_test', y_test)
np.save('ph2_z_test', z_test)

np.save('ph2_X_val', X_val)
np.save('ph2_y_val', y_val)
np.save('ph2_z_val', z_val)

np.save('ph2_X_train_reg', X_train_reg)
np.save('ph2_y_train_reg', y_train_reg)
np.save('ph2_z_train_reg', z_train_reg)

np.save('ph2_X_test_reg', X_test_reg)
np.save('ph2_y_test_reg', y_test_reg)
np.save('ph2_z_test_reg', z_test_reg)

np.save('ph2_X_val_reg', X_val_reg)
np.save('ph2_y_val_reg', y_val_reg)
np.save('ph2_z_val_reg', z_val_reg)