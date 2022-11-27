from pathlib import Path
import tempfile
import SimpleITK as sitk
from typing import Tuple
import numpy as np
from skimage import morphology


def read_raw_sitk(
    binary_file_path: Path, image_size: Tuple[int], sitk_pixel_type: int = sitk.sitkInt16,
    image_spacing: Tuple[float] = None, image_origin: Tuple[float] = None, big_endian: bool = False
) -> sitk.Image:
    """ Reads a raw binary scalar image.
    Args:
        binary_file_path (Path): Raw, binary image file path.
        image_size (Tuple): Size of image (e.g. (512, 512, 121))
        sitk_pixel_type (int, optional): Pixel type of data.
            Defaults to sitk.sitkInt16.
        image_spacing (Tuple, optional): Image spacing, if none given assumed
            to be [1]*dim. Defaults to None.
        image_origin (Tuple, optional): image origin, if none given assumed to
            be [0]*dim. Defaults to None.
        big_endian (bool, optional): Byte order indicator, if True big endian, else
            little endian. Defaults to False.
    Returns:
        (sitk.Image): Loaded image.
    """
    pixel_dict = {
        sitk.sitkUInt8: "MET_UCHAR",
        sitk.sitkInt8: "MET_CHAR",
        sitk.sitkUInt16: "MET_USHORT",
        sitk.sitkInt16: "MET_SHORT",
        sitk.sitkUInt32: "MET_UINT",
        sitk.sitkInt32: "MET_INT",
        sitk.sitkUInt64: "MET_ULONG_LONG",
        sitk.sitkInt64: "MET_LONG_LONG",
        sitk.sitkFloat32: "MET_FLOAT",
        sitk.sitkFloat64: "MET_DOUBLE",
    }
    direction_cosine = [
        "1 0 0 1",
        "1 0 0 0 1 0 0 0 1",
        "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1",
    ]

    dim = len(image_size)

    element_spacing = " ".join(["1"] * dim)
    if image_spacing is not None:
        element_spacing = " ".join([str(v) for v in image_spacing])

    img_origin = " ".join(["0"] * dim)
    if image_origin is not None:
        img_origin = " ".join([str(v) for v in image_origin])

    header = [
        ("ObjectType = Image\n").encode(),
        (f"NDims = {dim}\n").encode(),
        (f'DimSize = {" ".join([str(v) for v in image_size])}\n').encode(),
        (f"ElementSpacing = {element_spacing}\n").encode(),
        (f"Offset = {img_origin}\n").encode(),
        (f"TransformMatrix = {direction_cosine[dim - 2]}\n").encode(),
        (f"ElementType = {pixel_dict[sitk_pixel_type]}\n").encode(),
        ("BinaryData = True\n").encode(),
        ("BinaryDataByteOrderMSB = " + str(big_endian) + "\n").encode(),
        (f"ElementDataFile = {binary_file_path.resolve()}\n").encode(),
    ]
    fp = tempfile.NamedTemporaryFile(suffix=".mhd", delete=False)
    # Not using the tempfile with a context manager and auto-delete
    # because on windows we can't open the file a second time for ReadImage.
    fp.writelines(header)
    fp.close()

    img = sitk.ReadImage(str(fp.name))
    Path(fp.name).unlink()
    return img


def generate_lm_mask(landmarks: np.ndarray, img_size: Tuple, dilate: bool = False):
    """Generates a landmarks volume mask
    Args:
        landmarks (np.ndarray): [x,y,z] coordinates of the landmarks
        img_size (Tuple): size of the original image to match with the masks
        dilate (bool, optional): whether to dilate the lm pixels in order to
            see them in the visualization
    Returns:
        (np.ndarray): landmarks mask
    """
    lm_mask = np.zeros(img_size, dtype=np.uint8)
    lm_mask[landmarks[:, 0], landmarks[:, 1], landmarks[:, 2]] = 1
    if dilate:
        se = morphology.ball(5)
        lm_mask = morphology.binary_dilation(lm_mask, se)
    lm_mask = lm_mask * 255
    return lm_mask


def save_img_from_array(
    volume: np.ndarray, reference: sitk.Image, filepath: Path
):
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
