from nilearn.image import crop_img
from nilearn.masking import unmask


def _unmask_timeseries(timeseries, mask, crop=False):
    """Unmask timeseries, and reconstructs volumetric images, and optionally crop image.

    Parameters
    ----------
    timeseries : np.ndarray
        Extracted ROI timeseries (ts x voxels).
    mask : nib.Nifti1Image
        Mask image in NIFTI format.
    crop : bool, optional
        Crops the image to get rid of unnecessary blank spaces around the borders of brain, by default False.

    Returns
    -------
    4D np.ndarray
        Reconstructed volumetric images (ts x x_dim x y_dim x z_dim)
    """
    img = unmask(timeseries, mask)
    if crop:
        img = crop_img(img)
    return img.get_fdata().transpose(3, 0, 1, 2)
