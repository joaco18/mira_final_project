from neurite.tf.callbacks import ModelCheckpoint, ReduceLROnPlateau
from pystrum.pytools.plot import jitter
import matplotlib
# imports
import os, sys
# third party imports
import numpy as np
import tensorflow as tf
from scipy.ndimage import zoom
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset.copd_dataset import DirLabCOPD, vxm_data_generator_cache
from utils.metrics import target_registration_error

assert tf.__version__.startswith('2.'), 'This tutorial assumes Tensorflow 2.0+'
import voxelmorph as vxm
import neurite as ne

vol_shape = (256, 256, 128)
nb_features = [
    [16, 32, 32, 32],
    [32, 32, 32, 32, 32, 16, 16]
]
batch_size = 1

data_test = DirLabCOPD(
    cases=['all'],
    partitions=['train'],
    return_lm_mask=True,
    normalization_cfg=None
)

# build vxm network
vxm_model = vxm.networks.VxmDense(vol_shape, nb_features, int_steps=0)

sample_test = []
for i in tqdm(range(len(data_test))):
    sample = data_test[i]
    val_input = [
        sample['e_img'][np.newaxis, ...],
        sample['i_img'][np.newaxis, ...]
    ]

    val_pred = vxm_model.predict(val_input)
    pred_warp = val_pred[1]
    def_field = zoom(sample['i_img'], (2, 2, 1 / sample['i_img_factor']))

    data = [tf.convert_to_tensor(f, dtype=tf.float32) for f in [sample['i_landmark_pts'], def_field]]

    annotations_warped = vxm.utils.point_spatial_transformer(data)[0, ...].numpy()
    m, s = target_registration_error(sample['i_landmark_pts'], annotations_warped, sample['ref_metadata']['spacing'])
    print(f'{"-" * 10} {sample["case"]} {"-" * 10}')
    print(f'Provided displacement: {sample["disp_mean"]} | {sample["disp_std"]}')
    print(f'Computed displacement: {m} | {s}')

test_generator = vxm_data_generator_cache(data_test, batch_size=batch_size)
