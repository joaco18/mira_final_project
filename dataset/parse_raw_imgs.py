import SimpleITK as sitk
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import logging

from dataset import utils

logging.basicConfig(level=logging.INFO)

dirlab_meta = {
    'copd1': {'size': (512, 512, 121), 'spacing': (0.625, 0.625, -2.5), 'partition': 'train'},
    'copd2': {'size': (512, 512, 102), 'spacing': (0.645, 0.645, -2.5), 'partition': 'train'},
    'copd3': {'size': (512, 512, 126), 'spacing': (0.652, 0.652, -2.5), 'partition': 'train'},
    'copd4': {'size': (512, 512, 126), 'spacing': (0.590, 0.590, -2.5), 'partition': 'train'},
    'copd5': {'size': (512, 512, 131), 'spacing': (0.647, 0.647, -2.5), 'partition': 'val'},
    'copd6': {'size': (512, 512, 119), 'spacing': (0.633, 0.633, -2.5), 'partition': 'val'},
    'copd7': {'size': (512, 512, 112), 'spacing': (0.625, 0.625, -2.5), 'partition': 'test'},
    'copd8': {'size': (512, 512, 115), 'spacing': (0.586, 0.586, -2.5), 'partition': 'test'},
    'copd9': {'size': (512, 512, 116), 'spacing': (0.644, 0.644, -2.5), 'partition': 'test'},
    'copd10': {'size': (512, 512, 135), 'spacing': (0.742, 0.742, -2.5), 'partition': 'test'},
}


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
        for img_path, lm_path in zip([i_img_path, e_img_path], [ilm_path, elm_path]):
            img = utils.read_raw_sitk(
                img_path, meta['size'], sitk.sitkInt16, meta['spacing']
            )
            # flip vertical axis:
            # img_array = np.flip(sitk.GetArrayFromImage(img), axis=2)
            img_out_path = case_out_path / f'{img_path.stem}.nii.gz'
            # utils.save_img_from_array(img_array, img, img_out_path)
            sitk.WriteImage(img, img_out_path)

            # Generate landmarks mask
            landmarks = pd.read_csv(
                lm_path, header=None, sep='\t |\t', engine='python').values.astype(int)

            lm_mask = utils.generate_lm_mask(landmarks, meta['size'])
            lm_mask = np.moveaxis(lm_mask, [0, 1, 2], [2, 1, 0])

            lm_out_path = case_out_path / f'{img_path.stem}_lm.nii.gz'
            utils.save_img_from_array(lm_mask, img, lm_out_path)

        # Store the sample metadata
        row = [i_img_path, e_img_path, ilm_path, elm_path] + [meta['partition']] + \
            list(meta['size']) + list(meta['spacing'])
        df.append(row)
    columns = [
        'i_img_path', 'e_img_path', 'i_lm_path', 'e_lm_path', 'partition',
        'size_x', 'size_y', 'size_z', 'space_x', 'space_y', 'space_z'
    ]
    df = pd.DataFrame(df, columns=columns)
    df.to_csv(out_path/'dir_lab_copd.csv')


if __name__ == '__main__':
    main()
