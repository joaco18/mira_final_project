import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from pathlib import Path
from typing import List


def elastix_wrapper(
    fix_img_path: Path,
    mov_img_path: Path,
    res_path: Path,
    parameters_path: Path,
    keep_just_useful_files: bool = True
):
    """Wraps Elastix command line interface into a python function
    Args:
        fix_img_path (Path): Path to the fix image
        mov_img_path (Path): Path to the moving image
        res_path (Path): Path where to store the register image and transformation parameters
        parameters_path (Path): Path to the parameters map file
        keep_just_useful_files (bool, optional): Wheter to delete the rubish Elastix outputs.
            Defaults to True.
    Returns:
        (Path): Path where the transformation matrix is stored
    """
    # Fix filenames and create folders
    mov_img_name = mov_img_path.name.split(".")[0]
    if res_path.name.endswith('.img') or ('.nii' in res_path.name):
        res_filename = f'{res_path.name.split(".")[0]}.nii.gz'
        res_path = res_path.parent / 'res_tmp'
    else:
        res_filename = f'{mov_img_name}.nii.gz'
        res_path = res_path / 'res_tmp'
    res_path.mkdir(exist_ok=True, parents=True)

    # Run elastix
    subprocess.call([
        'elastix', '-out', str(res_path), '-f', str(fix_img_path), '-m',
        str(mov_img_path), '-p', str(parameters_path)
    ])

    # Fix resulting filenames
    (res_path/'result.0.nii.gz').rename(res_path.parent/res_filename)
    transformation_file_name = f'TransformParameters_{mov_img_name}.txt'
    (res_path/'TransformParameters.0.txt').rename(res_path.parent/transformation_file_name)

    if keep_just_useful_files:
        shutil.rmtree(res_path)

    return res_path.parent/transformation_file_name


def transformix_wrapper(
    mov_img_path: Path,
    res_path: Path,
    transformation_path: Path,
    keep_just_useful_files: bool = True
):
    """Wraps elastix command line interfase into a python function
    Args:
        mov_img_path (Path): Path to the moving image
        res_path (Path): Path where to store the register image and transformation parameters
        transformation_path (Path): Path to the transformation map file
        keep_just_useful_files (bool, optional): Wheter to delete the rubish Elastix outputs.
            Defaults to True.
    """
    # Fix filenames and create folders
    if res_path.name.endswith('.img') or ('.nii' in res_path.name):
        res_filename = f'{res_path.name.split(".")[0]}.nii.gz'
        res_path = res_path.parent / 'res_tmp'
    else:
        mov_img_name = mov_img_path.name.split(".")[0]
        res_filename = f'{mov_img_name}.nii.gz'
        res_path = res_path / 'res_tmp'
    res_path.mkdir(exist_ok=True, parents=True)

    # Run transformix
    subprocess.call([
        'transformix', '-out', str(res_path), '-in', str(mov_img_path),
        '-tp', str(transformation_path)
    ])

    # Fix resulting filenames
    (res_path/'result.nii.gz').rename(res_path.parent/res_filename)
    if keep_just_useful_files:
        shutil.rmtree(res_path)


def modify_field_parameter_map(
    field_value_list: List[tuple], in_par_map_path: Path, out_par_map_path: Path = None
):
    """Modifies the parameter including/overwriting the Field/Value pairs passed
    Args:
        field_value_list (List[tuple]): List of (Field, Value) pairs to modify
        in_par_map_path (Path): Path to the original parameter file
        out_par_map_path (Path, optional): Path to the destiny parameter file
            if None, then the original is overwritten. Defaults to None.
    """
    pm = sitk.ReadParameterFile(str(in_par_map_path))
    for [field, value] in field_value_list:
        pm[field] = (value, )
    out_par_map_path = in_par_map_path if out_par_map_path is None else out_par_map_path
    sitk.WriteParameterFile(pm, str(out_par_map_path))
