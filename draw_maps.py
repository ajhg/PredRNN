
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.utils as utils
import tensorflow.keras.backend as backend

from keras_custom.layers.STLSTM import *

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

########################################################################################################################

np.set_printoptions(precision=1)
np.set_printoptions(linewidth=110)

dataset = np.load('datasets/datatensor_7984_atl_s.npy')

dataset[:,:,:,:,0] *= 0.0001 # adequating range
dataset[:,:,:,:,1] *= 0.02 # adequating range
dataset[:,:,:,:,2] *= 0.02 # adequating range

# selecting one random sample
teste01 = dataset[8000,:,:,:,:]
#teste01 = dataset[8500,:,:,:,:]

## Data Spec ##
IMG_SIZE = 17
NUM_CHNLS = 3
INPUT_LEN = 10
PRED_LEN = 10
TOTAL_LEN = INPUT_LEN + PRED_LEN

## Model Spec ##
NUM_CELL = 4
FILTERS = 128
KERNEL_SIZE = 3

x = np.concatenate((teste01[:INPUT_LEN,:,:,:], np.zeros_like(teste01[INPUT_LEN:TOTAL_LEN,:,:,:])), axis=0)
x = np.expand_dims(x, axis=0) 

cells = StackedSTLSTMCells([STLSTMCell(filters=FILTERS, kernel_size=KERNEL_SIZE) for _ in range(NUM_CELL)])
predRNN = keras.Sequential([
  STLSTM2D(cells, return_sequences=True, input_shape=(TOTAL_LEN, IMG_SIZE, IMG_SIZE, NUM_CHNLS)),
  keras.layers.Reshape(target_shape=(IMG_SIZE*TOTAL_LEN, IMG_SIZE, FILTERS)),
  keras.layers.Conv2D(filters=NUM_CHNLS, kernel_size=1),
  keras.layers.Reshape(target_shape=(TOTAL_LEN, IMG_SIZE, IMG_SIZE, NUM_CHNLS))
  ])
#predRNN.summary()

optimizer = keras.optimizers.Adam(lr=0.001)

def l1_l2_loss(target, pred):
  diff = target - pred
  loss_ = tf.pow(diff, 2) + tf.abs(diff) # L2 + L1
  return backend.mean(loss_, axis=list(range(5)))

predRNN.compile(optimizer = optimizer, loss = l1_l2_loss, metrics = [tf.keras.metrics.mse])

predRNN.load_weights('parameters/7983_hgt_u_v_100epc.h5')

prediction = predRNN.predict([x])

# reversing range to original
prediction[:,:,:,:,0] *= 10000
prediction[:,:,:,:,1] *= 50
prediction[:,:,:,:,2] *= 50
teste01[:,:,:,0] *= 10000
teste01[:,:,:,1] *= 50
teste01[:,:,:,2] *= 50

# uninvert latitudes
pred_cor = np.zeros_like(prediction)
test_cor = np.zeros_like(teste01)
for lat in range(17):
  pred_cor[:,:,16-lat,:,:]=prediction[:,:,lat,:,:]
  test_cor[:,16-lat,:,:]=teste01[:,lat,:,:]

##############################

time_inst = 14

fig = plt.figure()

fig.suptitle('1979/1983 training data ##### 100 training epochs ##### sample 15/20 (30 hours ahead forecast)')

ax1 = fig.add_subplot(231)
im1 = ax1.imshow(test_cor[time_inst+1,:,:,1])
ax1.set_title('zonal wind, (reanalysis)')
ax1.axis('off')
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation='vertical')

ax2 = fig.add_subplot(232)
im2 = ax2.imshow(test_cor[time_inst+1,:,:,2])
ax2.set_title('meridional wind (reanalysis)')
ax2.axis('off')
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax, orientation='vertical')

ax3 = fig.add_subplot(233)
im3 = ax3.imshow(test_cor[time_inst+1,:,:,0])
ax3.set_title('geopotential height (reanalysis)')
ax3.axis('off')
divider = make_axes_locatable(ax3)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im3, cax=cax, orientation='vertical')

ax4 = fig.add_subplot(234)
ax4.imshow(pred_cor[0,time_inst,:,:,1])
ax4.set_title('zonal wind (forecast)')
ax4.axis('off')
divider = make_axes_locatable(ax4)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation='vertical')

ax5 = fig.add_subplot(235)
ax5.imshow(pred_cor[0,time_inst,:,:,2])
ax5.set_title('meridional wind (forecast)')
ax5.axis('off')
divider = make_axes_locatable(ax5)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax, orientation='vertical')

ax6 = fig.add_subplot(236)
ax6.imshow(pred_cor[0,time_inst,:,:,0])
ax6.set_title('geopotential height (forecast)')
ax6.axis('off')
divider = make_axes_locatable(ax6)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im3, cax=cax, orientation='vertical')

plt.show()

