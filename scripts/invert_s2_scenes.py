'''
Lookup-table based inversion of Sentinel-2 scenes (original S2 scenes in L2A processing
level, no uncertainty applied).

The inversion strategy is based on the median value of the 100 best performing solutions
in terms of the minimum root mean squared error (RMSE) between observed and simulated
spectra.
'''

import cv2
import numpy as np
import geopandas as gpd
import pandas as pd

from datetime import date
from eodal.config import get_settings
from eodal.core.band import Band
from eodal.core.raster import RasterCollection
from eodal.core.sensors import Sentinel2
from eodal.operational.mapping.sentinel2 import Sentinel2Mapper
from eodal.utils.sentinel2 import ProcessingLevels, get_S2_platform_from_safe
from eodal.operational.mapping import MapperConfigs
from pathlib import Path
from typing import List, Optional, Union

from rtm_imv.core.inversion import inv_img, retrieve_traits
from rtm_inv.core.lookup_table import generate_lut

logger = get_settings().logger

class RTMConfig:
    """
    Radiative transfer model inversion configuration set-up
    """
    def __init__(
            self,
            traits: List[str],
            rtm_params: Path,
            rtm: Optional[str] = 'prosail'
        ):
        """
        Class Constructor

        :param traits:
            list of traits to retrieve from the inversion process
            (depending on radiative transfer model chosen)
        :param rtm_params:
            CSV-file specifying RTM specific parameter ranges and their
            distribution
        :param rtm:
            name of the radiative transfer model to use. 'prosail' by
            default
        """
        self.traits = traits
        self.rtm_params = rtm_params
        self.rtm = rtm

        @property
        def rtm_params():
            return self.rtm_params


class LookupTableBasedInversion(RTMConfig):
    """
    Lookup-table based radiative transfer model inversion set-up
    """
    def __init__(
            self,
            n_solutions: int,
            cost_function: str,
            lut_size: int,
            method: Optional[str] = 'LHS',
            **kwargs
        ):
        """
        Class constructor

        :param n_solutions:
            number of solutions to use for lookup-table based inversion
        :param cost_function:
            name of the cost-function to evaluate similarity between
            observed and simulated spectra
        :param lut_size:
            size of the lookup-table (number of spectra to simulate)
        :param method:
            method to use for sampling the lookup-table
        :param kwargs:
            keyword arguments to pass to super-class
        """
        # call constructor of super-class
        super().__init__(**kwargs)
        self.n_solutions = n_solutions
        self.cost_function = cost_function
        self.method = method
        self.lut_size = lut_size

        @property
        def lut_size():
            return self.lut_size

def traits_from_s2(
        date_start: date,
        date_end: date,
        aoi: Union[Path, gpd.GeoDataFrame],
        rtm_config: RTMConfig,
        output_dir: Path,
        scene_cloud_cover_threshold: Optional[int] = 80,
        spatial_resolution: Optional[float] = 10.,
        resampling_method: Optional[int] = cv2.INTER_NEAREST_EXACT,
        processing_level: ProcessingLevels = ProcessingLevels.L2A,
        unique_feature_id: Optional[str] = None,
        **mapper_configs
    ): 
    """
    Retrieve traits from Sentinel-2 imagery using radiative transfer model
    inversion
    """
    # setup Sentinel-2 mapper to get the relevant scenes
    mapper_configs = MapperConfigs(
        spatial_resolution=spatial_resolution,
        resampling_method=resampling_method,
        **mapper_configs
    )

    # get a new mapper instance
    mapper = Sentinel2Mapper(
        date_start=date_start,
        date_end=date_end,
        processing_level=processing_level,
        cloud_cover_threshold=scene_cloud_cover_threshold,
        mapper_configs=mapper_configs,
        unique_id_attribute=unique_feature_id,
        feature_collection=aoi
    )

    mapper.get_scenes()
    s2_data = mapper.get_complete_timeseries()

    # extraction is based on features (1 to N field parcel geometries)
    features = mapper.feature_collection['features']

    # loop over features and perform inversion
    for idx, feature in enumerate(features):
        feature_id = mapper.get_feature_ids()[idx]
        feature_scenes = s2_data[feature_id]
        feature_metadata = mapper.observations[feature_id]
        feature_metadata['sensing_time'] = pd.to_datetime(feature_metadata.sensing_time)
        output_dir_feature = output_dir.joinpath(feature_id)
        output_dir_feature.mkdir(exist_ok=True)
        # loop over scenes
        for feature_scene in feature_scenes:
            # make sure we're looking at the right metadata
            metadata = feature_metadata[
                feature_metadata.sensing_time.dt.date == feature_scene.scene_properties.acquisition_time.date()
            ]
            # get viewing and illumination angles
            solar_zenith_angle = metadata['sun_zenith_angle'].iloc[0]
            solar_azimuth_angle = metadata['sun_azimuth_angle'].iloc[0]
            viewing_zenith_angle = metadata['sensor_zenith_angle'].iloc[0]
            viewing_azimuth_angle = metadata['sensor_azimuth_angle'].iloc[0]
            # get platform
            platform = get_S2_platform_from_safe(
                dot_safe_name=metadata['product_uri'].iloc[0]
            )
            # map to full platform name
            full_names = {'S2A': 'Sentinel2A', 'S2B': 'Sentinel2B'}
            platform = full_names[platform]

            # call function to generate lookup-table
            lut = generate_lut(
                sensor=platform,
                lut_params=rtm_config.rtm_params,
                solar_zenith_angle=solar_zenith_angle,
                viewing_zenith_angle=viewing_zenith_angle,
                solar_azimuth_angle=solar_azimuth_angle,
                viewing_azimuth_angle=viewing_azimuth_angle,
                lut_size=rtm_config.lut_size
            )

            band_names = feature_scene.band_names
            if 'SCL' in band_names: band_names.remove('SCL')

            # mask clouds and shadows if processing level is L2A
            if feature_scene.scene_properties.processing_level == ProcessingLevels.L2A:
                feature_scene.mask_clouds_and_shadows(
                    bands_to_mask=band_names, inplace=True
                )

            # invert the S2 scene by comparing ProSAIL simulated to S2 observed spectra
            s2_lut_spectra = lut[band_names].values
            s2_spectra = feature_scene.get_values(band_selection=band_names)
            if isinstance(s2_spectra, np.ma.MaskedArray):
                mask = s2_spectra.mask[0,:,:]
                s2_spectra = s2_spectra.data
            else:
                mask = np.zeros(shape=(s2_spectra.shape[1], s2_spectra.shape[2]), dtype='uint8')
                mask = mask.as_type('bool')
        
            logger.info(f'Feature {feature_id}: Starting inversion of {metadata.product_uri.iloc[0]}')
            lut_idxs = inv_img(
                lut=s2_lut_spectra,
                img=s2_spectra,
                mask=mask,
                cost_function=cost_function,
                n_solutions=n_solutions,
            )
            trait_img = retrieve_traits(
                lut=lut,
                lut_idxs=lut_idxs,
                traits=traits
            )

            # save traits to file
            trait_collection = RasterCollection()
            for tdx, trait in enumerate(traits):
                trait_collection.add_band(
                    Band,
                    geo_info=feature_scene['B02'].geo_info,
                    band_name=trait,
                    values=trait_img[tdx,:,:]
                )
            product_uri = metadata.product_uri.iloc[0]
            if product_uri.endswith('.SAFE'):
                product_uri = product_uri.replace('.SAFE', '')
            fname = f'{product_uri}_traits.tiff'
            trait_collection.to_rasterio(
                fpath_raster=output_dir_feature.joinpath(fname)
            )
            logger.info(f'Feature {feature_id}: Finished inversion of {metadata.product_uri.iloc[0]}')

if __name__ == '__main__':

    # output directory for writing trait images
    output_dir = Path(
        '/home/graflu/public/Evaluation/Projects/KP0031_lgraf_PhenomEn/02_Field-Campaigns/Satellite_Data'
    )

    # RTM configuration
    traits = ['lai']
    n_solutions = 100
    cost_function = 'rmse'
    rtm_params = Path('../parameters/prosail_s2.csv')
    lut_size = 50000

    lut_config = LookupTableBasedInversion(
        traits=traits,
        n_solutions=n_solutions,
        cost_function=cost_function,
        lut_size=lut_size,
        rtm_params=rtm_params
    )

    # S2 configuration
    # maximum scene-cloud cover
    scene_cloud_cover_threshold = 50.

    # define start and end of the time series
    date_start = date(2022,2,1)
    date_end = date(2022,7,1)

    # define area of interest
    area_of_interest = Path(
        '/home/graflu/public/Evaluation/Projects/KP0031_lgraf_PhenomEn/02_Field-Campaigns/Satellite_Data/bounding_box_strickhof_4326.geojson'
    )
    unique_feature_id='name'
    aoi = gpd.read_file(area_of_interest)
    aoi[unique_feature_id] = area_of_interest.name.split('.')[0]

    # spatial resolution of output product and spatial resampling method
    spatial_resolution = 10. # meters
    resampling_method = cv2.INTER_NEAREST_EXACT

    traits_from_s2(
        date_start=date_start,
        date_end=date_end,
        aoi=aoi,
        rtm_config=lut_config,
        scene_cloud_cover_threshold=scene_cloud_cover_threshold,
        spatial_resolution=spatial_resolution,
        resampling_method=resampling_method,
        unique_feature_id=unique_feature_id,
        output_dir=output_dir
    )
