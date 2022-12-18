from ants import ANTsImage
import ants
import time
import numpy as np
import pandas as pd

def ants_preprocess(img: ANTsImage, norm_min: float = 0.0, norm_max: float = 1.0, trunc_min: float = 0.05, trunc_max: float = 0.95):
    """
    Function to preprocess an ANTs image by applying normalization and intensity truncation

    Args:
        img (ANTsImage): the image to be preprocessed
        norm_min (float): minimum image intensity after normalization
        norm_max (float): maximum image intensity after normalization
        trunc_min (float): minimum image intensity after truncation
        trunc_max (float): maximum image intensity after truncation

    returns: preprocessed image
    """
    img_pp = ants.iMath(img, 'Normalize', norm_min, norm_max)
    img_pp = ants.iMath_truncate_intensity(img_pp, trunc_min, trunc_max)

    return img_pp


def ants_register(fixed: ANTsImage, moving:ANTsImage, transform: str = 'SyN', metric: str = 'meansquares', sampling: int = 32, iterations: tuple = None, res_path: str =None):
    """
    Function to perform registration with ANTsPy using different parameters
    Args:
        fixed (ANTsImage): fixed image 
        moving (ANTsImage): moving image 
        transform (str): type of transform to be fit to the images 
        metric (str): name of metric to be used in the optimization
        sampling (int): sampling parameter for the SyN transform (pixel neighborhood to be considered)
        iterations (tuple): max number of iterations in each resolution of the pyramid
        res_path (str): path to store the transforms 

    returns: the registration result and time 
    """
    start = time.time()
    mytx =ants.registration(fixed=fixed, 
                            moving=moving,
                            outprefix=str(res_path),
                            type_of_transform=transform,
                            syn_sampling= sampling,
                            syn_metric=metric,
                            reg_iterations=iterations,
                            total_sigma=0.25,
                            flow_sigma=10,
                            aff_iterations=(100, 100, 100, 50, 0),
                            aff_shrink_factors=(8, 6, 4, 2, 1),
                            aff_smoothing_sigmas=(10, 1, 0, 0, 0))

    reg_time = time.time() - start
    return mytx, reg_time

def ants_transform_landmarks (landmarks_fixed_df: pd.DataFrame, landmarks_moving_df: pd.DataFrame, transforms: list):
    """
    Function to apply ANTsTransform to Landmarks 
    Args:
        landmarks_fixed_df (dataframe): landmarks of the fixed image stored in a dataframe 
        landmarks_moving_df (dataframe): landmarks of the moving image stored in a dataframe 
        transforms (list): list of the paths to the transforms to be applied to the landmarks
    returns: registered landmarks and ground truth landmarks (fixed) stored in numpy arrays
    """
    landmarks_fixed_df = landmarks_fixed_df[['x', 'y','z']]
    landmarks_moving_df = landmarks_moving_df[['x', 'y','z']]

    landmarks_fixed = landmarks_fixed_df.to_numpy(dtype=np.float32)
    landmarks_result = ants.apply_transforms_to_points(3, landmarks_moving_df, transforms, whichtoinvert=[True, False]).to_numpy(dtype=np.float32)

    return landmarks_fixed, landmarks_result


