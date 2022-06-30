'''
Inverts a DataFrame of Planet-Scope SuperDove pixel observations
'''

import numpy as np
import geopandas as gpd
import pandas as pd

from eodal.config import get_settings
from pathlib import Path

from rtm_inv.core.config import RTMConfig, LookupTableBasedInversion
from rtm_inv.core.inversion import inv_img, retrieve_traits
from rtm_inv.core.lookup_table import generate_lut
from copy import deepcopy

logger = get_settings().logger

# define platform
platform = 'PlanetSuperDove'
# define scaling factor for reflectance values
gain = 0.0001

def traits_from_ps_pixels(
        ps_pixels: Path,
        rtm_config: RTMConfig
    ) -> gpd.GeoDataFrame: 
    """
    Retrieve traits from Planet SuperDove pixels stored in a GeoDataFrame
    using radiative transfer model inversion.

    :param ps_pixels:
        GeoDataFrame with PlanetScope pixel spectra (SuperDove sensor)
    :param rtm_config:
        radiative transfer model configuration (forward and inverse)
    :returns:
        GeoDataFrame with retrieved traits
    """

    pixels = gpd.read_file(ps_pixels)
    pixels['acquired'] = pd.to_datetime(pixels['acquired'])

    # loop over scenes and perform inversion per scene
    scenes = pixels.groupby(by='acquired')
    results = []
    for scene in scenes:

        logger.info(f'Working on {scene[0]}')

        data = deepcopy(scene[1])
        # get angles in the correct form
        # ProSAIL requires the solar zenith instead of the elevation
        data['sun_zenith'] = 90 - data['sun_elevation']
        # prepare viewing and illumination angles
        solar_zenith_angle = data['sun_zenith'].iloc[0]
        solar_azimuth_angle = data['sun_azimuth'].iloc[0]
        viewing_zenith_angle = data['view_angle'].iloc[0]
        viewing_azimuth_angle = data['satellite_azimuth'].iloc[0]

        # call function to generate lookup-table with simulated spectra
        lut = generate_lut(
            sensor=platform,
            lut_params=rtm_config.rtm_params,
            solar_zenith_angle=solar_zenith_angle,
            viewing_zenith_angle=viewing_zenith_angle,
            solar_azimuth_angle=solar_azimuth_angle,
            viewing_azimuth_angle=viewing_azimuth_angle,
            lut_size=rtm_config.lut_size
        )

        lut_spectra = lut[['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']].values

        # invert pixels of the scene
        sat_spectra = deepcopy(
            data[[
                'coastal_blue', 'blue', 'green_i', 'green', 'yellow', 'red', 'rededge', 'nir'
            ]]
        )
        # scale between 0 and 1
        sat_spectra *= gain
        # get spectra into array (n_bands, nrows, ncols)
        sat_spectra = sat_spectra.values.T
        sat_spectra = sat_spectra.reshape((sat_spectra.shape[0], sat_spectra.shape[1], 1))

        # mask is not required
        mask = np.ndarray(shape=sat_spectra.shape[1::], dtype='bool')
        mask[:,0] = False

        # inversion
        lut_idxs = inv_img(
            lut=lut_spectra,
            img=sat_spectra,
            mask=mask,
            cost_function=rtm_config.cost_function,
            n_solutions=rtm_config.n_solutions,
        )
        trait_img = retrieve_traits(
            lut=lut,
            lut_idxs=lut_idxs,
            traits=rtm_config.traits
        )

        # add traits to dataframe
        for rdx, trait in enumerate(rtm_config.traits):
            data[trait] = trait_img[rdx,:,0]
        results.append(data)

        logger.info(f'Finished {scene[0]}')

    return pd.concat(results) 

if __name__ == '__main__':

    # GeoPackage with Planet pixels
    data_dir = Path('/home/graflu/Documents/PlanetScope/PS_Eschikon_TS')
    ps_pixels = data_dir.joinpath('timeseries_BW_medians.gpkg')

    # RTM configuration
    traits = ['lai']
    n_solutions = 100
    cost_function = 'rmse'
    rtm_params = Path('../parameters/prosail_s2.csv')
    lut_size = 50000
    rtm_config = LookupTableBasedInversion(
        traits=traits,
        n_solutions=n_solutions,
        cost_function=cost_function,
        lut_size=lut_size,
        rtm_params=rtm_params
    )

    ps_pixels_traits = traits_from_ps_pixels(
        ps_pixels=ps_pixels,
        rtm_config=rtm_config
    )

    # save results to file
    fname = data_dir.joinpath('timeseries_BW_medians_lai.gpkg')
    ps_pixels_traits.to_file(fname, driver='GPKG')
