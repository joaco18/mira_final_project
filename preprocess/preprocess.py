import cv2
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


def get_lungs_mask_one_slice(img: np.ndarray, previous_mask: np.ndarray = None) -> np.ndarray:
    """Get the lungs mask in a single slice using threholding and several
    connected compoents computations
    Args:
        img (np.ndarray): image to find the lungs in it should be in houndsfield units.
        previous_mask (np.ndarray, optional): In computing the mask over the complete
            volume then the previous processed slice is used to constrain the results
            the following one. Defaults to None.
    Returns:
        np.ndarray: binary [0, 255] mask of the lungs
    """
    # Global thresholding
    mask = np.where(img < 600, 255, 0).astype('uint8')

    # If the image only contains background, return the empty mask
    if len(np.unique(mask)) == 1:
        return np.zeros_like(img).astype('uint8')

    # Get the connected componentes
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    # If the body of the patient occupies the full width, add a margin to
    # connect the upper and lower background connected components in one.
    shape = labels.shape
    border = np.ones((shape[0] + 4, shape[1] + 4)) * labels[0, 0]
    border[2:-2, 2:-2] = labels
    _, labels, stats, _ = cv2.connectedComponentsWithStats(border.astype('uint8'))

    # Ignore the CT scanner table and keep just the body.
    body = np.where(labels != labels[0, 0], 255, 0)
    _, b_labels, stats, _ = cv2.connectedComponentsWithStats(body.astype('uint8'))
    body = np.where(b_labels == np.argmax(stats[1:, -1]) + 1, 1, 0)

    # Only keep the structures inside the body
    labels = labels * body

    # If the mask from a previous slide exists, use it to contraint cases in
    # wich the trachea is bigger than the lungs or some othe bottom slices artifact
    if previous_mask is not None:
        # Get the connected components inside the body and make it the same size of
        # the padded labels image
        _, labels, stats, _ = cv2.connectedComponentsWithStats(labels.astype('uint8'))
        blank = np.zeros(labels.shape)
        blank[2:-2, 2:-2] = np.where(previous_mask != 0, 1, 0)
        previous_mask = blank.copy()

        # Select the labels inside the previous mask
        sel_labels = [i for i in np.unique(previous_mask * labels) if i != 0]
        labels_ = np.zeros_like(labels)
        for lab in sel_labels:
            labels_ = labels_ + np.where(labels == lab, lab, 0)
        labels = labels_.copy()

    # Keep just the two largest connected components inside the body
    _, labels, stats, _ = cv2.connectedComponentsWithStats(labels.astype('uint8'))
    idx = np.argsort(stats[1:, -1])[-2:] + 1
    if len(idx) != 0:
        lungs = np.where(labels == idx[0], 1, 0)
        if len(idx) > 1:
            lungs = lungs + np.where(labels == idx[1], 1, 0)
        # Perform whole filling with connected components
        lungs = np.where(lungs == 0, 255, 0)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(lungs.astype('uint8'))
        lungs = np.where(labels != labels[0, 0], 255, 0)
    else:
        lungs = labels.copy()
    # Remove the padding
    lungs = lungs[2:-2, 2:-2]
    return lungs


def get_lungs_mask(input_image: np.ndarray) -> np.ndarray:
    """Get the lungs mask in a chest CT scan using threholding and several
    connected compoents computations
    Args:
        img (np.ndarray): 3d volume to find the lungs in, should be in houndsfield units.
    Returns:
        np.ndarray: binary [0, 255] mask of the lungs
    """
    lungs = np.zeros_like(input_image)
    start = input_image.shape[0] // 2
    stops = [0, input_image.shape[0]]
    steps = [-1, 1]
    # Compute the slice segementation going from the middle slice to the top and bottom ones
    # in this way we can constrain the segementations and get rid of the trachea
    for stop, step in zip(stops, steps):
        prev = 0
        for k, i in enumerate(range(start, stop, step)):
            if k == 0:
                lungs[i, :, :] = get_lungs_mask_one_slice(input_image[i, :, :], None)
            else:
                lungs[i, :, :] = get_lungs_mask_one_slice(
                    input_image[i, :, :], lungs[prev, :, :])
            prev = i
    return lungs


def normalize_scan(scan, mask):
    mean = np.mean(scan[scan != 0])
    std = np.std(scan[scan != 0])
    scan = (scan - mean) / std
    return scan
