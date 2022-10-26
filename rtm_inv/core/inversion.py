'''
Actual inversion strategy using a pre-computed Lookup-Table (LUT)
and an image matrix of remotely sensed spectra.
'''

import numpy as np
import pandas as pd

from numba import njit, prange
from typing import List, Optional, Tuple

@njit(parallel=True, cache=True)
def inv_img(
        lut: np.ndarray,
        img: np.ndarray,
        mask: np.ndarray,
        cost_function: str,
        n_solutions: int
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        synthetic spectra. Currently implemented: 'rmse', 'mae',
        'contrast_function', 'squared_sum_of_differences'
    :param n_solutions:
        number of best solutions to return (where cost function is
        minimal)
    :returns:
        tuple with two ``np.ndarray`` of shape
        `(n_solutions, img_rows, img_columns)` where for each pixel
        the `n_solutions` best solutions are returned as row indices
        in the `lut` in the first tuple element and the corresponding
        cost function values in the second.
    """
    output_shape = (n_solutions, img.shape[1], img.shape[2])
    # array for storing best matching LUT indices
    lut_idxs = np.zeros(shape=output_shape, dtype='int32')
    # array for storing cost function values (required by some strategies)
    cost_function_values = np.zeros(shape=output_shape, dtype='float32')

    for row in prange(img.shape[1]):
        for col in range(img.shape[2]):
            # skip masked pixels
            if mask[row, col]:
                lut_idxs[:,row,col] = -1
                continue
            # get sensor spectrum (single pixel)
            image_ref = img[:,row,col]
            # image_ref_normalized = np.sum(np.abs(image_ref - (np.mean(image_ref))))
            # cost functions (from EnMap box) implemented in a
            # way Numba can handle them (cannot use keywords in numpy functions)
            delta = np.zeros(shape=(lut.shape[0],), dtype='float64')
            for idx in range(lut.shape[0]):
                if cost_function == 'rmse':
                    delta[idx] = np.sqrt(np.mean((image_ref - lut[idx,:])**2))
                elif cost_function == 'mae':
                    delta[idx] = np.sum(np.abs(image_ref - lut[idx,:]))
                elif cost_function == 'contrast_function':
                    delta[idx] = np.sum(
                        -np.log10(lut[idx,:] / image_ref) + lut[idx,:] / image_ref
                    )
                elif cost_function == 'squared_sum_of_differences':
                    delta[idx] = np.sum((lut[idx,:] - image_ref)**2)
            # find the smallest errors between simulated and observed spectra
            # we need the row index of the corresponding entries in the LUT
            delta_sorted = np.argsort(delta)
            lut_idxs[:,row,col] = delta_sorted[0:n_solutions]
            cost_function_values[:,row,col] = delta[delta_sorted[0:n_solutions]]
    return lut_idxs, cost_function_values

# @njit(cache=True, parallel=True)
def _retrieve_traits(
        trait_values: np.ndarray,
        lut_idxs: np.ndarray,
        cost_function_values: np.ndarray,
        measure: Optional[str] = 'Median'
    ) -> np.ndarray:
    """
    Uses the indices of the best matching spectra to retrieve the
    corresponding trait values from the LUT

    :param trait_values:
        array with traits entries in the LUT
    :param lut_idxs:
        indices of the best matching entries in the LUT
    :param cost_function_values:
        corresponding values of the cost function chosen
    :param measure:
        statistical measure to retrieve the solution per trait out
        of the *n* best solutions found before. Currently implemented
        are "median" (takes the median value of the best *n* solutions)
        and "weighted_mean" (mean weighted by the cost function values
        of the *n* best solutions)
    :returns:
        3-d image with trait values with shape (n_traits, nrows, ncols)
    """
    n_traits = trait_values.shape[1]
    n_solutions, rows, cols = lut_idxs.shape
    # allocate array for storing inversion results
    trait_img_shape = (n_traits, rows, cols)
    trait_img = np.zeros(trait_img_shape, dtype='float64')
    # loop over pixels and write inversion result to trait_img
    for row in prange(rows):
        for col in range(cols):
            # continue if the pixel was masked before and has therefore no
            # solutions
            if (lut_idxs[:,row,col] == -1).all():
                trait_img[:,row,col] = np.nan
                continue
            trait_vals_n_solutions = trait_values[lut_idxs[:,row,col],:]
            if measure == 'Median':
                trait_img[:,row,col] = \
                    np.median(trait_vals_n_solutions)
            elif measure == 'weighted_mean':
                denominator = np.sum(0.1 * cost_function_values[:,row,col])
                vest_sum = 0.
                for solution in range(n_solutions):
                    weight = 0.1 * cost_function_values[solution,row,col] / denominator
                    vest_sum = vest_sum + weight * trait_vals_n_solutions[solution]
                trait_img[:,row,col] = vest_sum[0]

    return trait_img

def retrieve_traits(
        lut: pd.DataFrame,
        lut_idxs: np.ndarray,
        cost_function_values: np.ndarray,
        traits: List[str],
        **kwargs
    ) -> np.ndarray:
    """
    Extracts traits from a lookup-table on results of `inv_img`

    :param lut:
        complete lookup-table from the RTM forward runs (i.e.,
        spectra + trait values) as ``pd.DataFrame``.
    :param lut_idxs:
        row indices in the `lut` denoting for each image pixel
        the *n* best solutions (smallest value of cost function
        between modelled and observed spectra)
    :param cost_function_values:
        corresponding values of the cost function chosen
    :param traits:
        name of traits to extract from the `lut`. The output
        array will have as many entries per pixel as traits.
    :param aggregation_function:
        name of the function to aggregate the *n* best solutions
        into a single final one. Calls [np.]median per default.
        Otherwise 'mean' can be passed.
    :param kwargs:
        further key-word arguments to pass to `_retrieve_traits`
    :returns:
        ``np.ndarray`` of shape `(len(traits), rows, cols)`
        where rows and columns are given by the shape of lut_idx
    """
    trait_values = lut[traits].values
    res_img = _retrieve_traits(
        trait_values=trait_values,
        lut_idxs=lut_idxs,
        cost_function_values=cost_function_values,
        **kwargs
    )
    return res_img
