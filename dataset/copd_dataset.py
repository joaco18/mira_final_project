from pathlib import Path
from typing import List
import pandas as pd
from skimage.exposure import match_histograms, equalize_adapthist
import SimpleITK as sitk
import numpy as np
from monai.transforms import Rand3DElasticd, Compose
from scipy.ndimage import zoom

import preprocess.preprocess as preproc
from utils import utils

data_path = Path().resolve().parent.parent / 'data'
# data_path = Path('../data')

rand_elastic = Rand3DElasticd(
    keys=["image", "label"],
    mode=("bilinear", "nearest"),
    prob=0.8,
    sigma_range=(5, 8),
    magnitude_range=(100, 200),
    spatial_size=(256, 256, 128),
    translate_range=(50, 50, 2),
    rotate_range=(np.pi / 36, np.pi / 36, np.pi),
    scale_range=(0.15, 0.15, 0.15),
    padding_mode="border",
)
train_transform = Compose([rand_elastic])

NORMALIZATION_CFG = {
    'norm_type': 'min-max',
    'mask': None,
    'max_val': 255,
    'window': [-1024, 600],
    'dtype': 'uint8',
}


class DirLabCOPD():
    def __init__(
            self,
            data_path: Path = data_path,
            cases: List[str] = ['all'],
            partitions: List[str] = ['train', 'val', 'test'],
            return_lm_mask: bool = False,
            normalization_cfg: dict = None,
            return_imgs: bool = True,
            return_lung_masks: bool = False,
            return_body_masks: bool = False,
            standardize_scan: bool = False,
            resize_shape: tuple = None,
            resize: bool = False,
            clahe: bool = False,
            histogram_matching: bool = False
    ):
        """
        Args:
            data_path (Path, optional): Path to 'data' folder. Defaults to data_path.
            cases (List[str], optional): List of a subset of cases e.g ['copd1'].
                Defaults to ['all'].
            partitions (List[str], optional): Partitions to include.
                Defaults to ['train', 'val', 'test'].
            return_lm_mask (bool, optional): Whether to return a mask with the landmarks
                for each image. Defaults to False.
            normalization_cfg (dict, optional): How to normalize the images.
                Defaults to None
            return_imgs (bool, optional): Whether to return the images. Defaults to True.
            return_lung_masks (bool, optional): Whether to return the lung_masks.
                Defaults to False.
            return_body_masks (bool, optional): Whether to return the body_masks.
                Defaults to False.
            standardize_scan (bool, optional): Whether to normalize the scan with marwan's min-max.
                Defaults to False.
            resize (bool, optional): Whether to zoom data to have data of shape 256,256,128.
                Defaults to False.
            resize_shape (tuple, optional): shape to resize to.
            clahe (bool, optional): Whether to apply CLAHE enhacement or not. Defaults to False,
            histogram_matching (bool, optional): Wether to apply histogram matching between
                inhale and exhale images. Defaults to False,
        """
        self.data_path = data_path
        self.cases = cases
        self.partitions = partitions
        self.return_lm_mask = return_lm_mask
        self.normalization_cfg = normalization_cfg
        
        # Adjsut normalization to uniform the codes
        if standardize_scan:
            if self.normalization_cfg is not None:
                self.normalization_cfg['norm_type'] = 'min-max-nobkgrd'
                self.normalization_cfg['mask'] = 'lungs' if self.return_lung_masks else None
                self.normalization_cfg['dtype'] = 'float32'
                self.normalization_cfg['max_val'] = 1.0
            else:
                self.normalization_cfg = {
                    'norm_type': 'min-max-nobkgrd', 'dtype': 'float32', 'max_val': 1.0,
                    'mask': 'lungs' if self.return_lung_masks else None
                }

        self.resize = resize
        self.return_imgs = return_imgs
        self.return_lung_masks = return_lung_masks
        self.return_body_masks = return_body_masks
        self.resize_shape = resize_shape

        self.clahe = clahe
        self.histogram_matching = histogram_matching

        # Read the dataset csv
        self.df = pd.read_csv(self.data_path / 'dir_lab_copd' / 'dir_lab_copd.csv', index_col=0)

        # Filter by cases selection
        if 'all' not in self.cases:
            self.filter_by_case_selection()
        self.filter_by_partitions()

    def filter_by_partitions(self):
        self.df = self.df.loc[self.df.partition.isin(self.partitions)]
        self.df.reset_index(drop=True, inplace=True)

    def filter_by_case_selection(self):
        self.df = self.df.loc[self.df.case.isin(self.cases)]
        self.df.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        df_row = self.df.loc[idx, :].squeeze()
        case_path = (self.data_path.parent / df_row.i_img_path).parent
        case = df_row.case

        # Create item container
        sample = {}
        sample['case'] = case

        sample['i_lung_mask'] = None
        sample['e_lung_mask'] = None

        # Load inhale data (fixed image)
        # Image
        sample['i_img_path'] = str(case_path / f'{case}_iBHCT.nii.gz')
        sample['i_full_mask_path'] = str(case_path / f'{case}_iBHCT_lungs.nii.gz')
        if self.return_imgs:
            sample['i_img'] = sitk.ReadImage(sample['i_img_path'])
            sample['ref_metadata'] = utils.extract_metadata(sample['i_img'])
            sample['i_img'] = sitk.GetArrayFromImage(sample['i_img'])
            sample['i_img'] = np.moveaxis(sample['i_img'], [0, 1, 2], [2, 1, 0])

        if self.return_lung_masks:
            sample['i_lung_mask'] = sitk.GetArrayFromImage(sitk.ReadImage(
                str(case_path / f'{case}_iBHCT_lungs.nii.gz')))
            sample['i_lung_mask'] = np.where(sample['i_lung_mask'] == 2, 255, 0)
            sample['i_lung_mask'] = np.moveaxis(sample['i_lung_mask'], [0, 1, 2], [2, 1, 0])

        if self.return_body_masks:
            sample['i_body_mask'] = sitk.GetArrayFromImage(sitk.ReadImage(
                str(case_path / f'{case}_iBHCT_lungs.nii.gz')))
            sample['i_body_mask'] = np.where(sample['i_body_mask'] != 0, 255, 0)
            sample['i_body_mask'] = np.moveaxis(sample['i_body_mask'], [0, 1, 2], [2, 1, 0])

        # Preprocess inhale:
        if self.normalization_cfg is not None:
            norm_cfg = self.normalization_cfg.copy()
            if isinstance(norm_cfg['mask'], str):
                if norm_cfg['mask'] == 'body':
                    norm_cfg['mask'] = sample['i_body_mask']
                elif norm_cfg['mask'] == 'lungs':
                    norm_cfg['mask'] = sample['i_lung_mask']
            sample['i_img'] = preproc.normalize(sample['i_img'], **norm_cfg)
        if self.clahe:
            sample['i_img'] = equalize_adapthist(sample['i_img'], (8, 8, 8), 2)

        # Landmarks
        if self.return_lm_mask:
            sample['i_landmark_mask'] = sitk.GetArrayFromImage(
                sitk.ReadImage(str(case_path / f'{case}_iBHCT_lm.nii.gz')))
            sample['i_landmark_mask'] = np.moveaxis(
                sample['i_landmark_mask'], [0, 1, 2], [2, 1, 0])
        sample['i_landmark_pts'] = pd.read_csv(
            case_path / f'{case}_300_iBH_xyz_r1.csv', header=None, index_col=None).values

        # Load exahale data (fixed image)
        # Image
        sample['e_img_path'] = str(case_path / f'{case}_eBHCT.nii.gz')
        sample['e_full_mask_path'] = str(case_path / f'{case}_eBHCT_lungs.nii.gz')
        if self.return_imgs:
            sample['e_img'] = sitk.ReadImage(str(case_path / f'{case}_eBHCT.nii.gz'))
            sample['e_img'] = sitk.GetArrayFromImage(sample['e_img'])
            sample['e_img'] = np.moveaxis(sample['e_img'], [0, 1, 2], [2, 1, 0])

        if self.return_lung_masks:
            sample['e_lung_mask'] = sitk.GetArrayFromImage(sitk.ReadImage(
                str(case_path / f'{case}_eBHCT_lungs.nii.gz')))
            sample['e_lung_mask'] = np.where(sample['e_lung_mask'] == 2, 255, 0)
            sample['e_lung_mask'] = np.moveaxis(sample['e_lung_mask'], [0, 1, 2], [2, 1, 0])

        if self.return_body_masks:
            sample['e_body_mask'] = sitk.GetArrayFromImage(sitk.ReadImage(
                str(case_path / f'{case}_eBHCT_lungs.nii.gz')))
            sample['e_body_mask'] = np.where(sample['e_body_mask'] != 0, 255, 0)
            sample['e_body_mask'] = np.moveaxis(sample['e_body_mask'], [0, 1, 2], [2, 1, 0])

        # Preprocess exhale:
        if self.normalization_cfg is not None:
            norm_cfg = self.normalization_cfg.copy()
            if isinstance(norm_cfg['mask'], str):
                if norm_cfg['mask'] == 'body':
                    norm_cfg['mask'] = sample['e_body_mask']
                elif norm_cfg['mask'] == 'lungs':
                    norm_cfg['mask'] = sample['e_lung_mask']
            sample['e_img'] = preproc.normalize(sample['e_img'], **norm_cfg)

        if self.clahe:
            sample['e_img'] = equalize_adapthist(sample['e_img'], (8, 8, 8), 2)

        if self.histogram_matching:
            sample['e_img'] = match_histograms(sample['e_img'], sample['i_img'])

        # Resize images
        if self.resize:
            # inhale
            factor = tuple(self.resize_shape[i] / sample['i_img'].shape[i] for i in range(3))
            sample['i_img'] = zoom(sample['i_img'], (0.5, 0.5, factor))
            sample['i_img_factor'] = factor
            if 'i_lung_mask' in sample.keys():
                sample['i_lung_mask'] = zoom(sample['i_lung_mask'], (0.5, 0.5, factor))
            if 'i_body_mask' in sample.keys():
                sample['i_body_mask'] = zoom(sample['i_body_mask'], (0.5, 0.5, factor))

            # exhale
            factor = tuple(self.resize_shape[i] / sample['e_img'].shape[i] for i in range(3))
            sample['e_img'] = zoom(sample['e_img'], (0.5, 0.5, factor))
            sample['e_img_factor'] = factor
            if 'e_lung_mask' in sample.keys():
                sample['e_lung_mask'] = zoom(sample['e_lung_mask'], (0.5, 0.5, factor))
            if 'e_body_mask' in sample.keys():
                sample['e_body_mask'] = zoom(sample['e_body_mask'], (0.5, 0.5, factor))

        # Landmarks
        if self.return_lm_mask:
            sample['e_landmark_mask'] = sitk.GetArrayFromImage(
                sitk.ReadImage(str(case_path / f'{case}_eBHCT_lm.nii.gz')))
            sample['e_landmark_mask'] = np.moveaxis(
                sample['e_landmark_mask'], [0, 1, 2], [2, 1, 0])
        sample['e_landmark_pts'] = pd.read_csv(
            case_path / f'{case}_300_eBH_xyz_r1.csv', header=None, index_col=None).values

        # Include baseline metrics
        baseline_keys = [
            'disp_mean', 'disp_std', 'observers_mean', 'observers_std', 'lowest_mean', 'lowest_std']
        for key in baseline_keys:
            sample[key] = df_row[key]

        return sample


def transform_sample(sample, transforms):
    # transform e_img or i_img by probability of 0.5
    sample_new = {}
    if np.random.rand() > 0.5:
        transformed_sample = transforms(
            {'image': sample['e_img'][np.newaxis, ...], 'label': sample['e_lung_mask'][np.newaxis, ...]})
        sample_new['e_img'] = transformed_sample['image'][0]
        sample_new['e_lung_mask'] = transformed_sample['label'][0]
        sample_new['i_img'] = sample['i_img']

    else:
        transformed_sample = transforms(
            {'image': sample['i_img'][np.newaxis, ...], 'label': sample['i_lung_mask'][np.newaxis, ...]})
        sample_new['i_img'] = transformed_sample['image'][0]
        sample_new['i_lung_mask'] = transformed_sample['label'][0]
        sample_new['e_img'] = sample['e_img']

    return sample_new


def vxm_data_generator_cache(samples, batch_size=32, transforms=None, use_labels=False):
    """
    Generator that takes in data of size [N, H, W, D], and yields data for
    our custom vxm model. Note that we need to provide numpy data for each
    input, and each output.

    inputs:  moving [bs, H, W,D, 1], fixed image [bs, H, W,D, 1]
    outputs: moved image [bs, H, W,D, 1], zero-gradient [bs, H, W,D, 2]
    """
    while True:
        idx1 = np.random.randint(0, len(samples), size=batch_size)

        chosen_samples = [transform_sample(samples[i], transforms) if transforms else samples[i] for i in idx1]

        moving_images = [sample['e_img'][np.newaxis, ..., np.newaxis] for sample in chosen_samples]
        fixed_images = [sample['i_img'][np.newaxis, ..., np.newaxis] for sample in chosen_samples]

        moving_images = np.concatenate(moving_images, axis=0)
        fixed_images = np.concatenate(fixed_images, axis=0)

        vol_shape = moving_images.shape[1:-1]  # extract data shape
        ndims = len(vol_shape)

        zero_phi = np.zeros((batch_size, *vol_shape, ndims))

        if use_labels:
            downsize = 2
            moving_masks = [
                (samples[i]['e_lung_mask'] / 255.0)[np.newaxis, ::downsize, ::downsize, ::downsize, np.newaxis] for i in
                idx1]
            fixed_masks = [
                (samples[i]['i_lung_mask'] / 255.0)[np.newaxis, ::downsize, ::downsize, ::downsize, np.newaxis] for i in
                idx1]
            moving_masks = np.concatenate(moving_masks, axis=0)
            fixed_masks = np.concatenate(fixed_masks, axis=0)
            inputs = [moving_images, fixed_images, moving_masks]
            outputs = [fixed_images, zero_phi, fixed_masks]
        else:
            inputs = [moving_images, fixed_images]
            outputs = [fixed_images, zero_phi]

        yield inputs, outputs
