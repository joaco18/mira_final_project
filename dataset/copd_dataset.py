from pathlib import Path
from typing import List
import pandas as pd
import SimpleITK as sitk
import numpy as np

from utils import utils

data_path = Path('__file__').resolve().parent.parent / 'data'


class DirLabCOPD():
    def __init__(
        self,
        data_path: Path = data_path,
        cases: List[str] = ['all'],
        partitions: List[str] = ['train', 'val', 'test'],
        return_lm_mask: bool = False
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
        """
        self.data_path = data_path
        self.cases = cases
        self.partitions = partitions
        self.return_lm_mask = return_lm_mask

        # Read the dataset csv
        self.df = pd.read_csv(data_path /'dir_lab_copd' / 'dir_lab_copd.csv', index_col=0)

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
        sample['i_img'] = sitk.ReadImage(str(case_path / f'{case}_iBHCT.nii.gz'))
        sample['ref_metadata'] = utils.extract_metadata(sample['i_img'])
        sample['i_img'] = sitk.GetArrayFromImage(sample['i_img'])
        sample['i_img'] = np.moveaxis(sample['i_img'], [0, 1, 2], [2, 1, 0])
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
        sample['e_img'] = sitk.ReadImage(str(case_path / f'{case}_eBHCT.nii.gz'))
        sample['e_img'] = sitk.GetArrayFromImage(sample['e_img'])
        sample['e_img'] = np.moveaxis(sample['e_img'], [0, 1, 2], [2, 1, 0])
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
