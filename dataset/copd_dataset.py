from pathlib import Path
from typing import List
import pandas as pd
import SimpleITK as sitk
import numpy as np

import preprocess.preprocess as preproc
from utils import utils

data_path = Path('__file__').resolve().parent / 'data'

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
        """
        self.data_path = data_path
        self.cases = cases
        self.partitions = partitions
        self.return_lm_mask = return_lm_mask
        self.normalization_cfg = normalization_cfg
        self.return_imgs = return_imgs
        self.return_lung_masks = return_lung_masks
        self.return_body_masks = return_body_masks

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

        # Load inhale data (fixed image)
        # Image
        sample['i_img_path'] = str(case_path / f'{case}_iBHCT.nii.gz')
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
                str(case_path / f'{case}_eBHCT_lungs.nii.gz')))
            sample['i_body_mask'] = np.where(sample['i_body_mask'] != 0, 255, 0)
            sample['i_body_mask'] = np.moveaxis(sample['i_body_mask'], [0, 1, 2], [2, 1, 0])

        if self.normalization_cfg is not None:
            sample['i_img'] = preproc.normalize(sample['i_img'], **self.normalization_cfg)

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

        if self.normalization_cfg is not None:
            sample['e_img'] = preproc.normalize(sample['e_img'], **self.normalization_cfg)
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
