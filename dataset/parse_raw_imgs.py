import SimpleITK as sitk
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import logging
import json

from dataset import data_utils
from utils import utils

logging.basicConfig(level=logging.INFO)


def main():
    # Define the datapath from the file
    data_path = Path(__file__).resolve().parent.parent
    out_path = data_path / 'data' / 'dir_lab_copd'
    data_path = data_path / 'data' / 'dir_lab_copd_raw'

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rip", dest="raw_images_path", help="Path to the dir_lab_copd_raw folder",
        default=str(data_path))
    parser.add_argument(
        "--op", dest="output_path", help="Directory where to store the result",
        default=str(out_path))
    args = parser.parse_args()

    data_path = Path(args.raw_images_path)
    out_path = Path(args.output_path)

    with open(str(data_path.parent / 'dir_lab_copd_metadata.json'), 'r') as json_file:
        dirlab_meta = json.load(json_file)

    df = []
    for case_path in sorted(data_path.iterdir()):
        # Define paths
        case = case_path.name
        logging.info(f'Processing case: {case}')
        ilm_path = case_path / f'{case}_300_iBH_xyz_r1.txt'
        i_img_path = case_path / f'{case}_iBHCT.img'
        elm_path = case_path / f'{case}_300_eBH_xyz_r1.txt'
        e_img_path = case_path / f'{case}_eBHCT.img'

        case_out_path = out_path / case
        case_out_path.mkdir(exist_ok=True, parents=True)

        # Get metadata:
        meta = dirlab_meta[case]

        # Parse raw image and parse landmarks
        img_out_paths, lm_out_paths, lm_pts_out_paths = [], [], []
        for img_path, lm_path in zip([i_img_path, e_img_path], [ilm_path, elm_path]):
            img = data_utils.read_raw_sitk(
                img_path, meta['size'], sitk.sitkInt16, meta['spacing'])
            # flip vertical axis:
            img_out_path = case_out_path / f'{img_path.stem}.nii.gz'
            sitk.WriteImage(img, img_out_path)

            # Generate landmarks mask
            landmarks = pd.read_csv(
                lm_path, header=None, sep='\t |\t', engine='python').astype('int')
            lm_pts_out_path = case_out_path / f'{lm_path.stem}.csv'
            landmarks.to_csv(lm_pts_out_path, index=False, header=False)
            landmarks = landmarks.values

            lm_mask = data_utils.generate_lm_mask(landmarks, meta['size'])
            lm_mask = np.moveaxis(lm_mask, [0, 1, 2], [2, 1, 0])

            lm_out_path = case_out_path / f'{img_path.stem}_lm.nii.gz'
            utils.save_img_from_array(lm_mask, img, lm_out_path)

            img_out_paths.append('/'.join(str(img_out_path).split('/')[-4:]))
            lm_out_paths.append('/'.join(str(lm_out_path).split('/')[-4:]))
            lm_pts_out_paths.append('/'.join(str(lm_pts_out_path).split('/')[-4:]))

        # Store the sample metadata
        metrics_keys = [
            'disp_mean', 'disp_std', 'observers_mean', 'observers_std', 'lowest_mean', 'lowest_std']
        row = img_out_paths + lm_out_paths + lm_pts_out_paths
        row = row + [meta['partition']] + list(meta['size']) + list(meta['spacing']) + [case]
        row = row + [meta[key] for key in metrics_keys]
        df.append(row)
    columns = [
        'i_img_path', 'e_img_path', 'i_lm_img_path', 'e_lm_img_path', 'i_lm_path', 'e_lm_path',
        'partition', 'size_x', 'size_y', 'size_z', 'space_x', 'space_y', 'space_z', 'case'
    ]
    columns = columns + metrics_keys
    df = pd.DataFrame(df, columns=columns)
    df.to_csv(out_path/'dir_lab_copd.csv')


if __name__ == '__main__':
    main()