import logging
import json
import time
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from dataset.copd_dataset import DirLabCOPD
from utils.metrics import target_registration_error
import elastix.elastix_utils as e_utils
from dataset.parse_raw_imgs import parse_raw_images
from dataset.generate_lung_masks import generate_lung_masks
import utils.utils as utils

logging.basicConfig(level=logging.INFO)
BASE_PATH = Path().resolve()


def run_experiment(
    dataset, param_maps_to_use, output_path, params_path, mask=None, experiment_name=None
):
    results = {}
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]

        # Define output paths
        pm = param_maps_to_use[-1]
        output_pm_path = output_path / pm.rstrip('.txt') / experiment_name
        output_pm_path.mkdir(exist_ok=True, parents=True)
        # Read and modify parameters file
        parameters_filename = params_path / pm
        result_path = output_pm_path / sample['case']
        result_path.mkdir(exist_ok=True, parents=True)

        # field_value_pairs = [('WriteResultImage', 'true'), ('ResultImageFormat', 'nii.gz')]
        field_value_pairs = [
            ('WriteResultImage', 'true'),
            ('ResultImageFormat', 'nii.gz')
        ]
        e_utils.modify_field_parameter_map(field_value_pairs, parameters_filename)

        # Create temporary image files with the preprocessings included and
        # also for the selected binary masks.
        res_path = result_path / 'res_tmp'
        res_path.mkdir(exist_ok=True, parents=True)

        param_maps_to_use_ = [str(params_path / p) for p in param_maps_to_use]
        # Inhale
        i_temp_path = res_path / 'i_img.nii.gz'
        utils.save_img_from_array_using_metadata(
            np.moveaxis(sample['i_img'], [0, 1, 2], [2, 1, 0]), sample['ref_metadata'], i_temp_path)
        i_body_mask_temp_path = res_path / 'i_body_mask_img.nii.gz'
        utils.save_img_from_array_using_metadata(
            np.moveaxis(sample['i_body_mask'], [0, 1, 2], [2, 1, 0]),
            sample['ref_metadata'], i_body_mask_temp_path)
        i_lungs_mask_temp_path = res_path / 'i_lungs_mask_img.nii.gz'
        utils.save_img_from_array_using_metadata(
            np.moveaxis(sample['i_lung_mask'], [0, 1, 2], [2, 1, 0]),
            sample['ref_metadata'], i_lungs_mask_temp_path)

        # Exhales
        e_temp_path = res_path / 'e_img.nii.gz'
        utils.save_img_from_array_using_metadata(
            np.moveaxis(sample['e_img'], [0, 1, 2], [2, 1, 0]), sample['ref_metadata'], e_temp_path)
        e_body_mask_temp_path = res_path / 'e_body_mask_img.nii.gz'
        utils.save_img_from_array_using_metadata(
            np.moveaxis(sample['e_body_mask'], [0, 1, 2], [2, 1, 0]),
            sample['ref_metadata'], e_body_mask_temp_path)
        e_lungs_mask_temp_path = res_path / 'e_lungs_mask_img.nii.gz'
        utils.save_img_from_array_using_metadata(
            np.moveaxis(sample['e_lung_mask'], [0, 1, 2], [2, 1, 0]),
            sample['ref_metadata'], e_lungs_mask_temp_path)

        # Register
        logging.info(f"Estimating transformation case {sample['case']}...")
        start = time.time()
        if mask is None:
            transform_map_path = e_utils.elastix_wrapper(
                i_temp_path, e_temp_path, res_path.parent, param_maps_to_use_,
                verbose=False, keep_just_useful_files=False
            )
        elif mask == 'body':
            transform_map_path = e_utils.elastix_wrapper(
                i_temp_path, e_temp_path, res_path.parent, param_maps_to_use_,
                i_body_mask_temp_path, e_body_mask_temp_path, verbose=False,
                keep_just_useful_files=False
            )
        else:
            transform_map_path = e_utils.elastix_wrapper(
                i_temp_path, e_temp_path, res_path.parent, param_maps_to_use_,
                i_lungs_mask_temp_path, e_lungs_mask_temp_path, verbose=False,
                keep_just_useful_files=False
            )
        reg_time = time.time()-start
        case_path = Path(sample['i_img_path']).parent

        name = f"{sample['case']}_300_iBH_xyz_r1.txt"
        lm_points_filepath = case_path / name
        # Correct transformation parameters file
        field_value_pairs = [
            ('ResultImageFormat', 'nii.gz'),
            ('ResultImagePixelType', "float"),
            ('FinalBSplineInterpolationorder', '0')
        ]
        e_utils.modify_field_parameter_map(field_value_pairs, transform_map_path)
        # Transform landmarks
        logging.info('Transforming points...')
        lm_out_filepath = res_path.parent / f'r_{name}'
        e_utils.transformix_wrapper(
            lm_points_filepath, lm_out_filepath, transform_map_path,
            points=True, verbose=False, keep_just_useful_files=False)

        # Transform lung_masks
        logging.info('Transforming lung mask...')
        full_mask_original_path = Path(sample['e_full_mask_path'])
        e_out_full_mask_temp_path = \
            e_lungs_mask_temp_path.parent.parent / f'r_{full_mask_original_path.name}'
        e_utils.transformix_wrapper(
            full_mask_original_path, e_out_full_mask_temp_path, transform_map_path,
            points=False, verbose=False, keep_just_useful_files=False)

        # Get transformed landmarks positions
        landmarks = pd.read_csv(lm_out_filepath, header=None, sep='\t |\t', engine='python')
        landmarks.columns = [
            'point', 'idx', 'input_index', 'input_point', 'ouput_index', 'ouput_point', 'def']
        landmarks_input = [lm[-4:-1] for lm in np.asarray(landmarks.input_index.str.split(' '))]
        landmarks_input = np.asarray(landmarks_input).astype('int')

        landmarks = utils.get_landmarks_array_from_txt_file(lm_out_filepath)

        # Get the TRE
        tre = target_registration_error(
            landmarks, sample['e_landmark_pts'], sample['ref_metadata']['spacing'])
        logging.info(f'TRE estimated: {tre[0]}, {tre[1]}')
        logging.info(f'Initial displacement GT: {sample["disp_mean"]}, {sample["disp_mean"]}')
        results[sample['case']] = {
            'mean_tre': tre[0],
            'std_tre': tre[1],
            'time': reg_time,
        }
    return results


def main():
    # Load configurations file
    logging.info('Reading configuration file...')
    cfg_path = BASE_PATH / 'elastix/inference_config.yaml'
    with open(cfg_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    # Parse the raw images
    if cfg['parsing_config']['doit']:
        logging.info('\nParsing raw images...')
        data_path = Path(cfg['parsing_config']['raw_images_path'])
        out_path = Path(cfg['parsing_config']['parsed_images_path'])
        df = parse_raw_images(data_path, out_path)
        df.to_csv(Path(cfg['parsing_config']['parsed_images_path'])/'dir_lab_copd.csv')

    # Extract the lungs mask
    if cfg['dataset']['extract_lung_masks']:
        logging.info('\nExtracting lungs masks images...')
        data = DirLabCOPD(
            data_path=Path(cfg['dataset']['data_path']),
            cases=cfg['dataset']['cases'],
            partitions=cfg['dataset']['partitions'],
            return_lm_mask=True,
            normalization_cfg=cfg['dataset']['normalization_cfg']
        )
        generate_lung_masks(data)

    # Run the registration
    logging.info('\nRegistering the images...')
    data = DirLabCOPD(
        data_path=Path(cfg['dataset']['data_path']),
        cases=cfg['dataset']['cases'],
        partitions=cfg['dataset']['partitions'],
        return_lm_mask=True,
        normalization_cfg=cfg['dataset']['normalization_cfg'],
        return_body_masks=cfg['dataset']['return_body_masks'],
        return_lung_masks=cfg['dataset']['return_lung_masks'],
        clahe=cfg['dataset']['clahe'],
        histogram_matching=cfg['dataset']['histogram_matching'],
    )

    output_path = Path(cfg['registration_config']['output_path'])
    results_path = output_path / 'inference_results.json'
    if results_path.exists():
        with open(results_path, 'r') as json_file:
            results = json.load(json_file)
    else:
        results = {}
        with open(results_path, 'w') as json_file:
            json.dump(results, json_file, indent=4, separators=(',', ': '))

    params_path = Path(cfg['registration_config']['params_path'])
    experiment_name = cfg['registration_config']['experiment_name']
    param_maps_to_use = cfg['registration_config']['param_maps_to_use']
    mask = Path(cfg['registration_config']['mask'])

    results[experiment_name] = run_experiment(
        data, param_maps_to_use, output_path, params_path, mask, experiment_name)

    logging.info('Experiment finished, storing the results...')
    with open(results_path, 'w') as json_file:
        json.dump(results, json_file, indent=4, separators=(',', ': '))


if __name__ == '__main__':
    main()
