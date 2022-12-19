
import numpy as np
from tqdm import tqdm
from pathlib import Path

from dataset.copd_dataset import DirLabCOPD
from preprocess.preprocess import get_lungs_mask
from utils.utils import save_img_from_array_using_metadata


def generate_lung_masks(data):
    # Define fixed imgage to use
    for i in tqdm(range(len(data))):
        sample = data[i]
        # print(f'{"-" * 5} Processing Case {sample["case"]} {"-" * 5}')
        # Extract and save lungs mask for inhale data
        i_img = np.moveaxis(sample['i_img'], [0, 1, 2], [2, 1, 0])
        i_lungs = get_lungs_mask(i_img)
        metadata = sample['ref_metadata']
        i_img_path = Path(sample['i_img_path'].replace('.nii.gz', '_lungs.nii.gz'))
        save_img_from_array_using_metadata(i_lungs.astype('uint8'), metadata, i_img_path)

        # Extract and save lungs mask for exhale data
        e_img = np.moveaxis(sample['e_img'], [0, 1, 2], [2, 1, 0])
        e_lungs = get_lungs_mask(e_img)
        metadata = sample['ref_metadata']
        e_img_path = Path(sample['e_img_path'].replace('.nii.gz', '_lungs.nii.gz'))
        save_img_from_array_using_metadata(e_lungs, metadata, e_img_path)


def main():
    data = DirLabCOPD(
        cases=['all'],
        partitions=['train', 'val', 'test'],
        return_lm_mask=True,
        normalization_cfg=None,
        data_path=Path('/home/kaouther/Documents/GitHub/mira_final_project/data')
    )
    generate_lung_masks(data)


if __name__ == '__main__':
    main()
