'''
Actual inversion strategy using a pre-computed Lookup-Table (LUT)
and an image matrix of remotely sensed spectra.
'''

import numpy as np
import pandas as pd

from numba import jit
from typing import List


@jit(nopython=True)
def inv_img(
        lut: np.ndarray,
        img: np.ndarray,
        mask: np.ndarray,
        cost_function: str,
        n_solutions: int
    ):
    """
    Lookup-table based inversion on images by minimizing a
    cost function using *n* best solutions to improve numerical
    robustness

    :param lut:
        LUT with synthetic (i.e., RTM-simulated) spectra in the
        spectral resolution of the sensor used. The shape of the LUT
        must equal (num_spectra, num_bands).
    :param img:
        image with sensor spectra. The number of spectral bands must
        match the number  of spectral bands in the LUT. The shape of
        the img must equal (num_bands, num_rows, num_columns).
    :param mask:
        mask of `img.shape[1], img.shape[2]` to skip pixels. If all
        pixels should be processed set all cells in `mask` to False.
    :param cost_function:
        cost function implementing similarity metric between sensor
        synthetic spectra. Currently implemented: 'rmse', 'mae', 'mNSE'
    :param n_solutions:
        number of best solutions to return (where cost function is
        minimal)
    :returns:
        ``np.ndarray`` of shape `(n_solutions, img_rows, img_columns)`
        where for each pixel the `n_solutions` best solutions are returned
        as row indices in the `lut`.
    """

    output_shape = (n_solutions, img.shape[1], img.shape[2])
    lut_idxs = np.zeros(shape=output_shape, dtype='uint32')

    for row in range(img.shape[1]):
        for col in range(img.shape[2]):

            # skip masked pixels
            if mask[row, col]:
                continue
            # get sensor spectrum (single pixel)
            image_ref = img[:,row,col]
            image_ref_normalized = np.sum(np.abs(image_ref - (np.mean(image_ref))))

            # cost functions (from EnMap box) implemented in a
            # way Numba can handle them (cannot use keywords in numpy functions)
            delta = np.zeros(shape=(lut.shape[0],), dtype='float64')
            for idx in range(lut.shape[0]):
                if cost_function == 'rmse':
                    delta[idx] = np.sqrt(np.mean((image_ref - lut[idx,:])**2))
                elif cost_function == 'mae':
                    delta[idx] = np.sum(np.abs(image_ref - lut[idx,:]))
                elif cost_function == 'mNSE':
                    delta[idx] = 1.0 - \
                        ((np.sum(np.abs(image_ref - lut[idx,:]))) / image_ref_normalized)

            # find the smallest errors between simulated and observed spectra
            # we need the row index of the corresponding entries in the LUT
            delta_sorted = np.argsort(delta)
            lut_idxs[:,row,col] = delta_sorted[0:n_solutions]
            # if row > 0 and col == 0 and row % 100 == 0:
            #     print(f'Processed Row {row}/{img.shape[1]}')

    return lut_idxs

@jit(nopython=True)
def _retrieve_traits(
        trait_values: np.ndarray,
        lut_idx: np.ndarray,
    ):
    n_traits = trait_values.shape[1]

    _, rows, cols = lut_idx.shape
    # allocate array for storing inversion results
    trait_img_shape = (n_traits, rows, cols)
    trait_img = np.zeros(trait_img_shape, dtype='float64')

    # loop over pixels and write inversion result to trait_img
    for trait in range(n_traits):
        for row in range(rows):
            for col in range(cols):
                trait_img[trait,row,col] = \
                    np.median(trait_values[lut_idx[:,row,col],trait])
    return trait_img

def retrieve_traits(
        lut: pd.DataFrame,
        lut_idx: np.ndarray,
        traits: List[str]
    ):
    """
    Extracts traits from a lookup-table on results of `inv_img`

    :param lut:
        complete lookup-table from the RTM forward runs (i.e.,
        spectra + trait values) as ``pd.DataFrame``.
    :param lut_idx:
        row indices in the `lut` denoting for each image pixel
        the *n* best solutions (smallest value of cost function
        between modelled and observed spectra)
    :param traits:
        name of traits to extract from the `lut`. The output
        array will have as many entries per pixel as traits.
    :param aggregation_function:
        name of the function to aggregate the *n* best solutions
        into a single final one. Calls [np.]median per default.
        Otherwise 'mean' can be passed.
    :returns:
        ``np.ndarray`` of shape `(len(traits), rows, cols)`
        where rows and columns are given by the shape of lut_idx
    """

    trait_values = lut[traits].values
    return _retrieve_traits(trait_values, lut_idx)
    
if __name__ == '__main__':

    from pathlib import Path
    from agrisatpy.core.band import Band
    from agrisatpy.core.raster import RasterCollection

    s2_lut = pd.read_csv('../../parameters/s2_prosail_demo.csv')
    s2_lut = s2_lut.dropna()

    traits = ['lai']

    fpath_raster = Path('/mnt/ides/Lukas/software/AgriSatPy/data/20190530_T32TMT_MSIL2A_S2A_pixel_division_10m.tiff')
    s2_data = RasterCollection.from_multi_band_raster(
        fpath_raster=fpath_raster
    )
    s2_bands = s2_data.band_names

    # synthetic spectra
    s2_lut_spectra = s2_lut[s2_bands].values
    s2_spectra = s2_data.get_values().astype(s2_lut_spectra.dtype)
    s2_spectra *= 0.0001

    # define mask so that all pixels are processed (i.e., no masking)
    mask = np.zeros(
        shape=(s2_spectra.shape[1], s2_spectra.shape[2]),
        dtype='uint8'
    )
    mask = mask.astype(bool)

    lut_idx = inv_img(
        lut=s2_lut_spectra,
        img=s2_spectra,
        mask=mask,
        cost_function='rmse',
        n_solutions=200,
    )
    trait_img = retrieve_traits(
        lut=s2_lut,
        lut_idx=lut_idx,
        traits=traits
    )

    collection = RasterCollection(
        Band,
        geo_info=s2_data['B02'].geo_info,
        band_name='LAI',
        values=trait_img[0,:,:]
    )
    fpath_out = Path('/mnt/ides/Lukas/03_Debug/LUT/20190530_T32TMT_MSIL2A_S2A_LAI-ProSAIL-rmse-200solutions.tif')
    collection.to_rasterio(fpath_raster=fpath_out)

    