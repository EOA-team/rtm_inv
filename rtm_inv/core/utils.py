"""
Title   : Resample spectral data by satellite spectral response function (SRF)
Creator : gperich & graflu
Date    : 26.10.2022
"""

import pandas as pd
import numpy as np

def resample_spectra(spectral_df: pd.DataFrame, sat_srf, wl_column: str = "WL"):
    """
    Function to spectrally resample hyperspectral 1nm RTM output to the
    spectral response function of a multi-spectral sensor.

    :param spectral_df: data frame containing the original spectra, needs to be in long-format with a
        "wavelength" (WL) column. Currently the WL column must be named the same as in the satellite SRF df.
    :param sat_srf: spectral response function (SRF) of the satellite. Must be in long-format with a WL column.
    :param wl_column: Name of the column containing the wavelength
    :return: A data frame with the resampled spectra
    """

    # get satellite bandnames
    sat_bands = sat_srf.drop(wl_column, axis=1).columns

    # get columns of the spectral_df (=individual spectra)
    indiv_spectra = spectral_df.drop(wl_column, axis=1).columns

    # force the satellite SRF onto the same spectra as the spectral DF
    innerdf = spectral_df.merge(sat_srf, on=wl_column)
    sat_srf = innerdf.loc[:, [wl_column, *sat_bands]]

    # iterate over every spectra
    out_list = []
    for spectrum in indiv_spectra:

        spec_df = spectral_df.loc[:, spectrum]

        # iterate over every satellite band
        spectrum_bandlist = []
        for satband in sat_bands:

            # get response of selected band
            sat_response = sat_srf.loc[:, satband]

            # multiply to get resampled (simulated) S2
            sim_vec = spec_df * sat_response

            # set zeroes to NA for calculations
            sim_vec[sim_vec == 0] = np.nan

            # return the selected quantile of the simulated S2 values
            sat_sim_refl = np.nansum(sim_vec)
            sat_sim_refl = sat_sim_refl / np.nansum(sat_response)

            spectrum_bandlist.append(sat_sim_refl)

        # append to DF / dict
        resampled_dict = dict(zip(sat_bands, spectrum_bandlist))

        # append to out_DF
        out_list.append(resampled_dict)

    out_df = pd.DataFrame.from_dict(out_list, orient="columns")
    out_df = out_df.transpose()
    out_df = out_df.reset_index()
    out_df = out_df.rename(columns={"index": "sat_bands"})

    return out_df

