import numpy as np
from typing import Tuple


def normalize(
    img: np.ndarray, norm_type: str = 'min-max', mask: np.ndarray = None,
    max_val: int = 255, window: Tuple[int] = (-1024, 600), dtype: str = None
):

    if window is not None:
        img = np.clip(img, window[0], window[1])
    if norm_type == 'min-max':
        img = min_max_norm(img, max_val, mask, dtype)
    else:
        raise Exception(
            f'Nomalization method {norm_type} not implemented try one of [min-max, ]'
        )
    return img


def min_max_norm(
    img: np.ndarray, max_val: int = None, mask: np.ndarray = None, dtype: str = None
) -> np.ndarray:
    """
    Scales images to be in range [0, 2**bits]

    Args:
        img (np.ndarray): Image to be scaled.
        max_val (int, optional): Value to scale images
            to after normalization. Defaults to None.
        mask (np.ndarray, optional): Mask to use in the normalization process.
            Defaults to None which means no mask is used.
        dtype (str, optional): Output datatype

    Returns:
        np.ndarray: Scaled image with values from [0, max_val]
    """
    if mask is None:
        mask = np.ones_like(img)

    # Find min and max among the selected voxels
    img_max = np.max(img[mask != 0])
    img_min = np.min(img[mask != 0])

    if max_val is None:
        # Determine possible max value according to data type
        max_val = np.iinfo(img.dtype).max

    # Normalize
    img = (img - img_min) / (img_max - img_min) * max_val
    img = np.clip(img, 0, max_val)

    # Adjust data type
    img = img.astype(dtype) if dtype is not None else img
    return img
