from pathlib import Path
import SimpleITK as sitk
import numpy as np


def save_img_from_array_using_referece(
    volume: np.ndarray, reference: sitk.Image, filepath: Path
) -> None:
    """Stores the volume in nifty format using the spatial parameters coming
        from a reference image
    Args:
        volume (np.ndarray): Volume to store as in Nifty format
        reference (sitk.Image): Reference image to get the spatial parameters from.
        filepath (Path): Where to save the volume.
    """
    # Save image
    if (type(volume) == list) or (len(volume.shape) > 3):
        if type(volume[0]) == sitk.Image:
            vol_list = [vol for vol in volume]
        else:
            vol_list = [sitk.GetImageFromArray(vol) for vol in volume]
        joiner = sitk.JoinSeriesImageFilter()
        img = joiner.Execute(*vol_list)
    else:
        img = sitk.GetImageFromArray(volume)
    img.SetDirection(reference.GetDirection())
    img.SetOrigin(reference.GetOrigin())
    img.SetSpacing(reference.GetSpacing())
    for key in reference.GetMetaDataKeys():
        img.SetMetaData(key, reference.GetMetaData(key))
    sitk.WriteImage(img, str(filepath))


def save_img_from_array_using_metadata(
    volume: np.ndarray, metadata: dict, filepath: Path
) -> None:
    """Stores the volume in nifty format using the spatial parameters coming
        from a reference image
    Args:
        volume (np.ndarray): Volume to store as in Nifty format
        metadata (dict): Metadata from the reference image to store the volumetric image.
        filepath (Path): Where to save the volume.
    """
    # Save image
    if (type(volume) == list) or (len(volume.shape) > 3):
        if type(volume[0]) == sitk.Image:
            vol_list = [vol for vol in volume]
        else:
            vol_list = [sitk.GetImageFromArray(vol) for vol in volume]
        joiner = sitk.JoinSeriesImageFilter()
        img = joiner.Execute(*vol_list)
    else:
        img = sitk.GetImageFromArray(volume)
    img.SetDirection(metadata['direction'])
    img.SetOrigin(metadata['origin'])
    img.SetSpacing(metadata['spacing'])
    for key, val in metadata['metadata'].items():
        img.SetMetaData(key, val)
    sitk.WriteImage(img, filepath)


def extract_metadata(img: sitk.Image) -> dict:
    """Extracts the useful metadata from the SimpleITK image to later use
    as referencee when storing another image e.g. an overlapping mask
    Args:
        img (sitk.Image): image from which to extract the metadas
    Returns:
        dict: metadata dictionary
    """
    header = {
        'direction': img.GetDirection(),
        'origin': img.GetOrigin(),
        'spacing': img.GetSpacing(),
        'metadata': {}
    }
    for key in img.GetMetaDataKeys():
        header['metadata'][key] = img.GetMetaData(key)
    return header
