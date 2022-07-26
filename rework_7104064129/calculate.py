import binary_loss
import Otsu

import tensorflow as tf
import numpy as np
import scipy.stats as stat

dataset = "ph1"
group = 1
where = "train"

#mse
dmos = np.load('{}_im{}_dmos_{}.npy'.format(dataset, group, where))
mse_dmos = np.load('{}_mse_im{}_dmos_{}.npy'.format(dataset, group, where))

mse_result = np.load('{}_im{}_MSE_result_{}.npy'.format(dataset, group, where))
# print(dmos.shape)
# print(mse_result.shape)

ground = Otsu.minmax_inverse(dmos, mse_dmos)
pred = Otsu.minmax_inverse(dmos, mse_result)

sub = []
for i in range(mse_result.shape[0]):
    sub.append((ground[i]-pred[i])[0])
    
#do someting on sub

#binary
bin_result = np.load('{}_im{}_regressor_result_{}.npy'.format(dataset, group, where))
bin_result_de = []
bin_sub = []
for i in range(bin_result.shape[0]):
    bin_result_de.append(binary_loss.bin2de(bin_result[i]))
    bin_sub.append((dmos[i]-binary_loss.bin2de(bin_result[i])-15*(group-1))[0])

bin_result_de = np.array((bin_result_de)).reshape(-1, 1)
PLCC_bin = stat.pearsonr(bin_result_de, dmos)
SROCC_bin = stat.spearmanr(bin_result_de, dmos)

PLCC_mse = stat.pearsonr(pred, ground)
SROCC_mse = stat.spearmanr(pred, ground)