
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

dataset = np.load('datasets/datatensor_7984_atl_s.npy')

dataset[:,:,:,:,0] *= 0.0001 # adequating range
dataset[:,:,:,:,1] *= 0.02 # adequating range
dataset[:,:,:,:,2] *= 0.02 # adequating range

## Data Spec ##
IMG_SIZE = np.shape(dataset)[-2]
NUM_CHNLS = np.shape(dataset)[-1]
INPUT_LEN = 10
PRED_LEN = 10
TOTAL_LEN = INPUT_LEN + PRED_LEN

TOTAL_SAMPLES = np.shape(dataset)[0]
NUM_TEST_SAMPLES = 1464
NUM_TRAIN_SAMPLES = TOTAL_SAMPLES - NUM_TEST_SAMPLES

#########################################################################
## Model Spec ##
NUM_CELL = 4
FILTERS = 128
KERNEL_SIZE = 3

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

############################################################
############################################################

### previsoes em lote para conferir erros

erros_zonal = np.zeros(NUM_TEST_SAMPLES)
erros_merid = np.zeros(NUM_TEST_SAMPLES)
for i in range(NUM_TEST_SAMPLES):
  teste = dataset[NUM_TRAIN_SAMPLES + i,:,:,:,:]
  x = np.concatenate((teste[:INPUT_LEN,:,:,:], np.zeros_like(teste[INPUT_LEN:TOTAL_LEN,:,:,:])), axis=0)
  x = np.expand_dims(x, axis=0) 
  pred = predRNN.predict([x])
  # reversing range to original
  pred[:,:,:,:,0] *= 10000
  pred[:,:,:,:,1] *= 50
  pred[:,:,:,:,2] *= 50
  teste[:,:,:,0] *= 10000
  teste[:,:,:,1] *= 50
  teste[:,:,:,2] *= 50
  erroz = 0
  errom = 0
  for j in range(10):
    erroz += np.sum( np.abs( pred[:,10+j,:,:,1] - teste[10+j,:,:,1] ) )
    errom += np.sum( np.abs( pred[:,10+j,:,:,2] - teste[10+j,:,:,2] ) )
  erros_zonal[i] = erroz
  erros_merid[i] = errom

erro_zonal = np.sum(erros_zonal)
print('\nzonal error, total')
print(erro_zonal)

erro_merid = np.sum(erros_merid)
print('\nmeridional error, total')
print(erro_merid)

erro_zonal_frame = (erro_zonal/NUM_TEST_SAMPLES)*0.1 # 10 frames per exemple
print('\nzonal error, per frame')
print(erro_zonal_frame)

erro_merid_frame = (erro_merid/NUM_TEST_SAMPLES)*0.1 # 10 frames per exemple
print('\nmeridional error, per frame')
print(erro_merid_frame)

erro_zonal_pixel = erro_zonal_frame/(IMG_SIZE**2) # imgsize*imgzise pixels per frame
print('\nzonal error, per pixel')
print(erro_zonal_pixel)

erro_merid_pixel = erro_merid_frame/(IMG_SIZE**2) # imgsize*imgzise pixels per frame
print('\nmeridional error, per pixel')
print(erro_merid_pixel)

print('')


