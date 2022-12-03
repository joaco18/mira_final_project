# from neurite.tf.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from pystrum.pytools.plot import jitter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib
# imports
import os, sys
# third party imports
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset.copd_dataset import DirLabCOPD, vxm_data_generator_cache

assert tf.__version__.startswith('2.'), 'This tutorial assumes Tensorflow 2.0+'
import voxelmorph as vxm
import neurite as ne

vol_shape = (256, 256, 128)
nb_features = [
    [16, 32, 32, 32],
    [32, 32, 32, 32, 32, 16, 16]
]
batch_size = 1

data_train = DirLabCOPD(
    cases=['all'],
    partitions=['train'],
    return_lm_mask=True,
    normalization_cfg=None,
    standardize_scan=True,
    resize=True,
    return_body_masks=True
)
sample_train = []
for i in tqdm(range(len(data_train))):
    sample_train.append(data_train[i])

train_generator = vxm_data_generator_cache(sample_train, batch_size=batch_size)

in_sample, out_sample = next(train_generator)
# visualize
print(in_sample[1].max())
_ = plt.hist(in_sample[1].reshape(-1), bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram with 'auto' bins")
plt.show()


images = [img[0, :, :, 81, 0] for img in in_sample + out_sample]
titles = ['moving', 'fixed', 'moved ground-truth (fixed)', 'zeros']
ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)


# build vxm network
vxm_model = vxm.networks.VxmDense(vol_shape, nb_features, int_steps=0)
steps_per_epoch = len(sample_train) // batch_size

# losses and loss weights
losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]
loss_weights = [1, 0.01]
cross_val_n = 0
save_freq = 1

# vxm_model.load_weights('brain_3d.h5')

model_save_callback = ModelCheckpoint(f'./vxmp/cross_val{cross_val_n}' + '/model_{epoch:02d}.hdf5',
                                      save_freq=save_freq * steps_per_epoch, save_weights_only=True, monitor='loss',
                                      mode='min', verbose=1)

reduce_on_plateau = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=1, min_delta=1e-4, mode='min')

# vxm_model.load_weights('brain_3d.h5')
vxm_model.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights)

hist = vxm_model.fit(train_generator, epochs=100, steps_per_epoch=steps_per_epoch, verbose=1,
                     callbacks=[model_save_callback, reduce_on_plateau])

