'''
Created on Oct 4, 2022

@author: graflu
'''

from pathlib import Path

from rtm_inv.core.lookup_table import generate_lut

if __name__ == '__main__':

    # set up lookup-table configurations
    rtm_params = Path('../parameters/prosail_danner-etal.csv')
    lut_size = 1000
    sampling_method = 'lhs' # latin-hypercube; alternative: frs - fully random sampling
    platform = 'Sentinel2A'

    # viewing and illumination angles (these are just randomly selected numbers, here)
    solar_zenith_angle = 46.3
    solar_azimuth_angle = 115.1
    viewing_zenith_angle = 3.4
    viewing_azimuth_angle = 76.9

    # path to spectral response function of Sentinel-2
    fpath_srf = Path('/home/graflu/public/Evaluation/Satellite_data/Sentinel-2/Documentation/S2_Specsheets/S2-SRF_COPE-GSEG-EOPG-TN-15-0007_3.1.xlsx')

    # generate the lookup-table - depending on the size of it this might take a while
    # the result is a pandas DataFrame with leaf and canopy parameters + ProSAIL spectra
    lut_srf = generate_lut(
        sensor=platform,
        lut_params=rtm_params,
        solar_zenith_angle=solar_zenith_angle,
        viewing_zenith_angle=viewing_zenith_angle,
        solar_azimuth_angle=solar_azimuth_angle,
        viewing_azimuth_angle=viewing_azimuth_angle,
        lut_size=lut_size,
        sampling_method=sampling_method,
        fpath_srf=fpath_srf
    )

    # lut_no_srf = generate_lut(
    #     sensor=platform,
    #     lut_params=rtm_params,
    #     solar_zenith_angle=solar_zenith_angle,
    #     viewing_zenith_angle=viewing_zenith_angle,
    #     solar_azimuth_angle=solar_azimuth_angle,
    #     viewing_azimuth_angle=viewing_azimuth_angle,
    #     lut_size=lut_size,
    #     sampling_method=sampling_method,
    #     fpath_srf=None
    # )
    #
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # plt.style.use('ggplot')
    #
    # cols_bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
    # f, ax = plt.subplots(nrows=2, ncols=5, figsize=(30,10))
    # ax = ax.flatten()
    # for idx, col_band in enumerate(cols_bands): 
    #     sns.kdeplot(x=lut_srf[col_band], y=lut_no_srf[col_band], ax=ax[idx], fill=True)
    #     ax[idx].set_xlabel('Reflectance with S2A SRF')
    #     ax[idx].set_ylabel('Reflectance with Gaussian SRF')
    #     ax[idx].set_title(col_band)
    # f.savefig(fpath_srf.parent.joinpath('effects_of_srf.png'), bbox_inches='tight')
