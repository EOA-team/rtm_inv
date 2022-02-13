'''
Generates a lookup table (LUT) with RTM forward runs
'''

import pandas as pd
from pathlib import Path
from typing import Optional

from agrisatpy.metadata.sentinel2 import parse_s2_scene_metadata
from agrisatpy.utils.sentinel2 import get_S2_platform_from_safe

from rtm_inv.lookup_table import LookupTable
from rtm_inv.rtm_adapter import RTM

def generate_lut(
        platform: str,
        lut_params: pd.DataFrame,
        lut_size: Optional[int] = 50000,
        rtm_name: Optional[str] = 'prosail',
        sampling_method: Optional[str] = 'LHS',
        solar_zenith_angle: Optional[float] = None,
        viewing_zenith_angle: Optional[float] = None,
        solar_azimuth_angle: Optional[float] = None,
        viewing_azimuth_angle: Optional[float] = None
    ) -> pd.DataFrame:

    # overwrite angles in LUT DataFrame
    lut_params['tts'] = solar_zenith_angle
    lut_params['tto'] = viewing_zenith_angle
    # calculate relative azimuth (psi)
    psi = abs(solar_azimuth_angle - viewing_azimuth_angle)
    lut_params['psi'] = psi

    # get input parameter samples first
    lut = LookupTable(param=lut_params)
    lut.generate_samples(lut_size, sampling_method)

    # and run the RTM in forward mode in the second step
    # outputs get resampled to the spectral resolution of the sensor
    rtm = RTM(lut=lut, rtm=rtm_name)
    lut_simulations = rtm.simulate_spectra(sensor='Sentinel2A')
    return lut_simulations

if __name__ == '__main__':

    # define lookup-table parameter ranges and distributions
    path_lut_params = Path('/mnt/ides/Lukas/software/rtm_inv/parameters/prosail_s2.csv')
    lut_params = pd.read_csv(path_lut_params)
    
    in_dir_safe = Path(
        '/mnt/ides/Lukas/03_Debug/Sentinel2/S2A_MSIL2A_20171213T102431_N0206_R065_T32TMT_20171213T140708.SAFE'
    )
    # get viewing and illumination angles
    scene_metadata, _ = parse_s2_scene_metadata(in_dir=in_dir_safe)
    solar_zenith_angle = scene_metadata['SUN_ZENITH_ANGLE']
    solar_azimuth_angle = scene_metadata['SUN_AZIMUTH_ANGLE']
    viewing_zenith_angle = scene_metadata['SENSOR_ZENITH_ANGLE']
    viewing_azimuth_angle = scene_metadata['SENSOR_AZIMUTH_ANGLE']
    # get platform
    platform = get_S2_platform_from_safe(dot_safe_name=in_dir_safe)
    # map to full plafform name
    full_names = {'S2A': 'Sentinel2A', 'S2B': 'Sentinel2B'}
    platform = full_names[platform]

    # call function to generate lookup-table
    lut = generate_lut(
        sensor=platform,
        lut_params=lut_params,
        solar_zenith_angle=solar_zenith_angle,
        viewing_zenith_angle=viewing_zenith_angle,
        solar_azimuth_angle=solar_azimuth_angle,
        vieiwing_azimuth_angle=viewing_azimuth_angle
    )
