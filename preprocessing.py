from argparse import Namespace

import nibabel as nib
import numpy as np

def crop(arr: np.array, args: Namespace) -> np.ndarray:
    """
    Crop a 3D array using the limits provided in args.crop_min and args.crop_max
    :param arr: array to be cropped
    :param args: Namespace.
    :return: cropped array.
    """
    return arr[
        args.crop_min[0]:args.crop_max[0],
        args.crop_min[1]:args.crop_max[1],
        args.crop_min[2]:args.crop_max[2]
    ]

def scale(arr: np.ndarray) -> np.ndarray:
    """
    Simple [0, 1] scaling.
    :param arr: array to be scaled.
    :return: scaled array.
    """
    min_value = arr[0, 0, 0]
    max_value = np.max(arr)
    return np.clip((arr - min_value) / (max_value - min_value), 0, 1)

def preprocess_mri(data_path: str, mask_path: str, output_path: str, args: Namespace):
    mri_scan = np.squeeze(nib.load(data_path).get_fdata())
    mask = np.squeeze(nib.load(mask_path).get_fdata())

    assert len(np.unique(mask)) == 2
    mask = mask / mask.max()
    mri_data = np.einsum("ijk -> jki", mri_scan)[::-1, ::-1, :]
    mri_data = mri_data * mask

    mri_data = crop(mri_data)
    mri_data = scale(mri_data)

    mri_data = mri_data.astype("float32")
    nib.save(nib.Nifti1Image(mri_data, None), output_path)


def preprocess_pet(data_path: str, output_path: str, args: Namespace):
    pet_data = np.squeeze(nib.load(data_path).get_fdata())

    pet_data = crop(pet_data)
    pet_data = scale(pet_data)

    pet_data = pet_data.astype("float32")
    nib.save(nib.Nifti1Image(pet_data, None), output_path)


def preprocess(namespace: Namespace) -> None:
    # TODO: add data preprocessing step. Read data files from disk, save preprocessed files to disk.
    pass
