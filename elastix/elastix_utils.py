import shutil
import subprocess
import SimpleITK as sitk
from pathlib import Path
from typing import List


def elastix_wrapper(
    fix_img_path: Path,
    mov_img_path: Path,
    res_path: Path,
    parameters_paths: List[Path],
    fix_mask_path: Path = None,
    mov_mask_path: Path = None,
    keep_just_useful_files: bool = True,
    verbose: bool = False
):
    """Wraps Elastix command line interface into a python function
    Args:
        fix_img_path (Path): Path to the fix image
        mov_img_path (Path): Path to the moving image
        res_path (Path): Path where to store the register image and transformation parameters
        fix_mask_path (Path, optional): Path to the fix image mask.
            Defaults to None which means no mask is used
        mov_mask_path (Path, optional): Path to the moving image mask.
            Defaults to None which means no mask is used
        parameters_paths (List[Path]): List of Paths to the parameters map file to use
        keep_just_useful_files (bool, optional): Wheter to delete the rubish Elastix outputs.
            Defaults to True.
        verbose (bool, optional): Wheter to the print the logs of elastix
            Defaults to False.
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
    if (fix_mask_path is not None) and (mov_mask_path is not None):
        command = [
            'elastix', '-out', str(res_path), '-f', str(fix_img_path), '-m', str(mov_img_path),
            '-fMask', str(fix_mask_path), '-mMask', str(mov_mask_path)
        ]
    elif (fix_mask_path is not None):
        command = [
            'elastix', '-out', str(res_path), '-f', str(fix_img_path), '-m', str(mov_img_path),
            '-fMask', str(fix_mask_path)
        ]
    elif (mov_mask_path is not None):
        command = [
            'elastix', '-out', str(res_path), '-f', str(fix_img_path), '-m', str(mov_img_path),
            '-mMask', str(mov_mask_path)
        ]
    else:
        command = [
            'elastix', '-out', str(res_path), '-f', str(fix_img_path), '-m', str(mov_img_path)]

    for i in parameters_paths:
        command.extend(['-p', i])

    if verbose:
        print(command)
        subprocess.call(command)
    else:
        subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    n = len(parameters_paths)-1
    # n = 1
    # Fix resulting filenames
    (res_path/f'result.{n}.nii.gz').rename(res_path.parent/res_filename)
    transformation_file_name = f'TransformParameters_{mov_img_name}.txt'
    shutil.copyfile(
        str(res_path/f'TransformParameters.{n}.txt'), str(res_path.parent/transformation_file_name))

    if keep_just_useful_files:
        shutil.rmtree(res_path)

    return res_path.parent/transformation_file_name


def transformix_wrapper(
    mov_img_path: Path,
    res_path: Path,
    transformation_path: Path,
    keep_just_useful_files: bool = True,
    points: bool = False,
    verbose: bool = False
):
    """Wraps elastix command line interfase into a python function
    Args:
        mov_img_path (Path): Path to the moving image
        res_path (Path): Path where to store the register image and transformation parameters
        transformation_path (Path): Path to the transformation map file
        keep_just_useful_files (bool, optional): Wheter to delete the rubish Elastix outputs.
            Defaults to True.
        points (bool, optional): Wheter to the things to transform are points or img
            Defaults to False.
        verbose (bool, optional): Wheter to the print the logs of elastix
            Defaults to False.
    """
    # Fix filenames and create folders
    if points:
        if res_path.name.endswith('.txt'):
            res_filename = f'{res_path.name.split(".")[0]}.txt'
            res_path = res_path.parent / 'res_tmp'
        else:
            mov_img_name = mov_img_path.name.split(".")[0]
            res_filename = f'{mov_img_name}.txt'
            res_path = res_path / 'res_tmp'
        res_path.mkdir(exist_ok=True, parents=True)
    else:
        if res_path.name.endswith('.img') or ('.nii' in res_path.name):
            res_filename = f'{res_path.name.split(".")[0]}.nii.gz'
            res_path = res_path.parent / 'res_tmp'
        else:
            mov_img_name = mov_img_path.name.split(".")[0]
            res_filename = f'{mov_img_name}.nii.gz'
            res_path = res_path / 'res_tmp'
        res_path.mkdir(exist_ok=True, parents=True)

    # Run transformix
    command = ['transformix', '-out', str(res_path), '-tp', str(transformation_path)]
    if points:
        command.extend(['-def', str(mov_img_path)])
    else:
        command.extend(['-in', str(mov_img_path)])

    if verbose:
        print(command)
        subprocess.call(command)
    else:
        subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Fix resulting filenames
    if points:
        shutil.copyfile(str(res_path/'outputpoints.txt'), str(res_path.parent/res_filename))
    else:
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
        if isinstance(value, list):
            pm[field] = (val for val in value)
        else:
            pm[field] = (value, )
    out_par_map_path = in_par_map_path if out_par_map_path is None else out_par_map_path
    sitk.WriteParameterFile(pm, str(out_par_map_path))
