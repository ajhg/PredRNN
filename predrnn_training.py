
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.utils as utils
import tensorflow.keras.backend as backend

from keras_custom.layers.STLSTM import *

#from pathlib import Path
import os
import time

#model_creation_device = '/cpu:0'
model_creation_device = '/gpu:0'

## Training Data Spec ##
IMG_SIZE = 17
NUM_CHNLS = 3
TOTAL_SAMPLES = 8768 - 20 # 1979 to 1984
NUM_TRAIN_SAMPLES = TOTAL_SAMPLES - 1464 # 1979 to 1983
INPUT_LEN = 10
PRED_LEN = 10
TOTAL_LEN = INPUT_LEN + PRED_LEN

## Model Spec ##
NUM_CELL = 4
FILTERS = 128
KERNEL_SIZE = 3

## Training Setup ##
#NUM_GPU = 4

## Data Feed Option ##
SHUFFLE_BUFFER_SIZE = NUM_TRAIN_SAMPLES
BATCH_SIZE = 4
EPOCHS = 25
BATCHES_PER_EPOCH = NUM_TRAIN_SAMPLES//BATCH_SIZE
BATCHES_PER_EPOCH_VALID = (TOTAL_SAMPLES - NUM_TRAIN_SAMPLES)//BATCH_SIZE

def input_fn():
  dataset = np.load('datasets/datatensor_7984_atl_s.npy')
  dataset[:,:,:,:,0] *= 400 # adequating range for better ANN performance
  dataset[:,:,:,:,1] *= 0.02 # adequating range
  dataset[:,:,:,:,2] *= 0.02 # adequating range

  x = np.concatenate((dataset[:NUM_TRAIN_SAMPLES,:INPUT_LEN,:,:,:], np.zeros_like(dataset[:NUM_TRAIN_SAMPLES,INPUT_LEN:TOTAL_LEN,:,:,:])), axis=1)
  y = dataset[:NUM_TRAIN_SAMPLES,1:,:,:,:]

  features = tf.data.Dataset.from_tensor_slices(x)
  features.prefetch(BATCH_SIZE)
  labels = tf.data.Dataset.from_tensor_slices(y)
  labels.prefetch(BATCH_SIZE)

  train_dataset = tf.data.Dataset.zip((features, labels))
  labels.prefetch(BATCH_SIZE)
  train_dataset = train_dataset.apply(tf.contrib.data.shuffle_and_repeat(SHUFFLE_BUFFER_SIZE, EPOCHS))
  labels.prefetch(BATCH_SIZE)
  train_dataset = train_dataset.map(lambda x,y: (backend.cast(x, 'float32'), backend.cast(y, 'float32')), num_parallel_calls=24)
  train_dataset = train_dataset.batch(BATCH_SIZE)
  train_dataset = train_dataset.prefetch(1)
  
  x_val = np.concatenate((dataset[NUM_TRAIN_SAMPLES:,:INPUT_LEN,:,:,:], np.zeros_like(dataset[NUM_TRAIN_SAMPLES:,INPUT_LEN:TOTAL_LEN,:,:,:])), axis=1)
  y_val = dataset[NUM_TRAIN_SAMPLES:,1:,:,:,:]

  features_val = tf.data.Dataset.from_tensor_slices(x_val)
  labels_val = tf.data.Dataset.from_tensor_slices(y_val)
  valid_dataset = tf.data.Dataset.zip((features_val, labels_val))
  valid_dataset = valid_dataset.repeat(EPOCHS)
  valid_dataset = valid_dataset.batch(BATCH_SIZE)
  valid_dataset = valid_dataset.prefetch(1)

  return train_dataset, valid_dataset

def l1_l2_loss(target, pred):
  diff = target - pred
  loss_ = tf.pow(diff, 2) + tf.abs(diff) # L2 + L1
  return backend.mean(loss_, axis=list(range(5)))

with keras.utils.custom_object_scope({'StackedSTLSTMCells':StackedSTLSTMCells,
                                      'STLSTMCell':STLSTMCell}): # Custom object scope for custom keras layers
  
  with tf.device(model_creation_device):
    cells = StackedSTLSTMCells([STLSTMCell(filters=FILTERS, kernel_size=KERNEL_SIZE) for _ in range(NUM_CELL)])
    predRNN = keras.Sequential([
      STLSTM2D(cells, return_sequences=True, input_shape=(TOTAL_LEN, IMG_SIZE, IMG_SIZE, NUM_CHNLS)),
      keras.layers.Reshape(target_shape=(IMG_SIZE*TOTAL_LEN, IMG_SIZE, FILTERS)),
      keras.layers.Conv2D(filters=NUM_CHNLS, kernel_size=1),
      keras.layers.Reshape(target_shape=(TOTAL_LEN, IMG_SIZE, IMG_SIZE, NUM_CHNLS))
      ])
    predRNN.summary()

  #predRNN_multi = utils.multi_gpu_model(predRNN, gpus=NUM_GPU) # Make Multi GPU model.
  predRNN_multi = predRNN
  #predRNN_multi.allow_growth = True
  #predRNN_multi.per_process_gpu_memory_fraction=0.1
  optimizer = keras.optimizers.Adam(lr=0.001)
  predRNN_multi.compile(optimizer = optimizer,
                        loss = l1_l2_loss,
                        #metrics = [tf.keras.metrics.mse])
                        metrics = ['mean_absolute_error', 'accuracy'])
  
  train_dataset, valid_dataset = input_fn() # Make TF datasets

  # load parameters from a previous training step
  predRNN.load_weights('parameters/7983_hgt_u_v_75epc.h5')

  # start training
  predRNN_multi.fit(train_dataset,
                    epochs=EPOCHS,
                    verbose=0,
                    validation_data=valid_dataset,
                    steps_per_epoch=BATCHES_PER_EPOCH,
                    validation_steps=BATCHES_PER_EPOCH_VALID)

  # save parameters
  predRNN_multi.save_weights('parameters/7983_hgt_u_v_100epc.h5', overwrite=True)

