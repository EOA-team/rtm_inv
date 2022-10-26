'''
Created on Oct 4, 2022

@author: graflu
'''

from pathlib import Path

from rtm_inv.core.lookup_table import generate_lut

if __name__ == '__main__':

    # set up lookup-table configurations
    rtm_params = Path('../parameters/prosail_lhc.csv')
    lut_size = 10000
    sampling_method = 'lhs' # latin-hypercube; alternative: frs - fully random sampling
    platform = 'Sentinel2A'
    # viewing and illumination angles (these are just randomly selected numbers, here)
    solar_zenith_angle = 46.3
    solar_azimuth_angle = 115.1
    viewing_zenith_angle = 3.4
    viewing_azimuth_angle = 76.9

    # generate the lookup-table - depending on the size of it this might take a while
    # the result is a pandas DataFrame with leaf and canopy parameters + ProSAIL spectra
    lut = generate_lut(
        sensor=platform,
        lut_params=rtm_params,
        solar_zenith_angle=solar_zenith_angle,
        viewing_zenith_angle=viewing_zenith_angle,
        solar_azimuth_angle=solar_azimuth_angle,
        viewing_azimuth_angle=viewing_azimuth_angle,
        lut_size=lut_size,
        sampling_method=sampling_method
    )

    lut.info()
    