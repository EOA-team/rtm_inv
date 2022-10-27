"""
"""

import pandas as pd
import numpy as np

from scipy.stats import poisson

green_peak_threshold = 547 # nm
green_region = (500, 600) # nm

def resample_spectra(spectral_df: pd.DataFrame, sat_srf: pd.DataFrame, wl_column: str = "WL"
                     ) -> pd.DataFrame:
    """
    Function to spectrally resample hyperspectral 1nm RTM output to the
    spectral response function of a multi-spectral sensor.

    :param spectral_df:
        data frame containing the original spectra, needs to be in long-format with a
        "wavelength" (WL) column. Currently the WL column must be named the same as
        in the satellite SRF df.
    :param sat_srf:
        spectral response function (SRF) of the satellite. Must be in long-format with
        a WL column.
    :param wl_column:
        Name of the column containing the wavelength
    :returns:
        A data frame with the resampled spectra
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
        # iterate over every target (multi-spectral) band
        spectrum_bandlist = []
        for satband in sat_bands:

            # get response of selected band
            sat_response = sat_srf.loc[:, satband]
            # start the resampling process. First, multiply the 1nm RTM
            # output with the spectral response function of the target
            # band
            sim_vec = spec_df * sat_response
            # set zeroes to NA for calculations
            sim_vec[sim_vec == 0] = np.nan

            # second, sum up the rescaled reflectance values and divide
            # them by the integral of the SRF coefficients
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

def chlorophyll_carotiniod_constraint(lut_df: pd.DataFrame) -> pd.DataFrame:
    """
    Samples leaf carotenoid content based on leaf chlorophyll content
    using a truncated Poisson distribution based on empirical bounds
    from the ANGERS dataset as proposed by Wocher et al. (2020,
    https://doi.org/10.1016/j.jag.2020.102219).

    :param lut_df:
        DataFrame with (uncorrelated) leaf chlorophyll a+b (cab)
        and leaf carotenoid (car) samples for running PROSAIL
    :returns:
        LUT with car values sampled based on car
    """
    def lower_boundary(cab: float | np.ndarray) -> float | np.ndarray:
        """
        lower boundary of the cab-car relationship reported by
        Wocher et al, 2020 based on the ANGERS dataset
        """
        return 0.223 / 4.684 * 3 * cab

    def upper_boundary(cab: float | np.ndarray) -> float | np.ndarray:
        """
        upper boundary of the cab-car relationship reported by
        Wocher et al, 2020 based on the ANGERS dataset
        """
        return 0.223 * 4.684 / 3 * cab + 2 * 0.986

    def cab_car_regression(cab: float | np.ndarray) -> float | np.ndarray:
        """
        empirical regression line between leaf chlorophyll and carotinoid
        content based on the ANGERS dataset reported by Wocher et al. (2020)
        """
        return 0.223 * cab + 0.986

    def sample_poisson(cab: float | np.ndarray) -> np.ndarray:
        """
        Sample leaf carotenoid values based on leaf chlorophyll
        values
        """
        if isinstance(cab, float):
            cab = [cab]
        car_samples = []
        for cab_val in cab:
            lower = lower_boundary(cab_val)
            upper = upper_boundary(cab_val)
            regression_val = cab_car_regression(cab_val)
            while True:
                car_sample = poisson.rvs(mu=regression_val)
                if lower < car_sample < upper:
                    car_samples.append(car_sample)
                    break
        return np.array(car_samples)

    # sample car based on cab as suggested by Wocher et al. (2020)
    cab = lut_df['cab']
    car_samples = sample_poisson(cab)
    # update the car column in the LUT
    out_df = lut_df.copy()
    out_df['car'] = car_samples
    return out_df
            
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # plt.style.use('seaborn-darkgrid')
    #
    # cab_linspace = np.linspace(cab.min(), cab.max(), cab.shape[0])
    # lower_constraint = lower_boundary(cab_linspace)
    # upper_constraint = upper_boundary(cab_linspace)
    # regression_line = cab_car_regression(cab_linspace)
    # f, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
    # sns.kdeplot(x='cab', y='car', data=lut_df, fill=True, ax=ax[0])
    # sns.kdeplot(x=cab, y=car_samples, fill=True, ax=ax[1])
    # ax[0].set_title(f'Original PROSAIL LUT (N={cab.shape[0]})')
    # ax[1].set_title(f'Redistributed PROSAIL LUT (N={cab.shape[0]})')
    # for idx in range(2):
    #     ax[idx].set_xlabel(r'Cab (PROSAIL) [$\mu g$ $cm^{-2}$]')
    #     ax[idx].set_ylabel(r'Car (PROSAIL) [$\mu g$ $cm^{-2}$]')
    #     ax[idx].plot(cab_linspace, lower_constraint, label='lower constraint')
    #     ax[idx].plot(cab_linspace, upper_constraint, label='upper constraint')
    #     ax[idx].plot(cab_linspace, regression_line, linestyle='dashed', label='Cab-Car Regression')
    # ax[1].legend()
    # ax[1].set_xlim(0,80)
    # ax[1].set_ylim(0,30)
    # plt.show()

def green_is_valid(wvls: np.ndarray, spectrum: np.ndarray) -> bool:
    """
    Checks if a simulated spectrum is valid in terms of the position
    of its green-peak. Green peaks occuring at wavelengths >547nm are
    consider invalid according to Wocher et al. (2020,
    https://doi.org/10.1016/j.jag.2020.102219)

    :param wvls:
        array with wavelengths in nm
    :param spectrum:
        corresponding spectral data
    :returns:
        `True` if the green-peak is valid, else `False`.
    """
    # get array indices of wavelengths in "green" part of the spectrum
    green_wvls_idx = np.where(wvls == green_region[0])[0][0], np.where(wvls == green_region[1])[0][0]
    green_spectrum = spectrum[green_wvls_idx[0]:green_wvls_idx[1]]
    green_wvls = wvls[green_wvls_idx[0]:green_wvls_idx[1]]
    green_peak = green_wvls[np.argmax(green_spectrum)]
    # green peaks smaller than the threshold are considered invalid
    if green_peak < green_peak_threshold:
        return False
    else:
        return True

    # import matplotlib.pyplot as plt
    # plt.style.use('seaborn-darkgrid')
    # plt.plot(wvls[green_wvls_idx[0]:green_wvls_idx[1]], green_spectrum)
    # plt.vlines(green_peak_threshold, label='Green Peak Threshold', ymin=0, ymax=0.07,
    #            color='green', linewidth=2)
    # plt.vlines(green_peak, label='Green Peak of Spectrum', ymin=0, ymax=0.07,
    #            color='grey', linewidth=2, linestyle='dashed')
    # plt.xlabel('Wavelength [nm]')
    # plt.ylabel('Reflectance Factors [%]')
    # plt.legend()
    # plt.ylim(0,0.07)
    # plt.show()
    
    
    
