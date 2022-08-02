'''
Lookup-table based inversion of Sentinel-2 scenes (original S2 scenes in L2A processing
level, no uncertainty applied).

The inversion strategy is based on the median value of the 100 best performing solutions
in terms of the minimum root mean squared error (RMSE) between observed and simulated
spectra.
'''

import cv2
import itertools
import numpy as np
import geopandas as gpd
import pandas as pd
import warnings

from datetime import date
from eodal.config import get_settings
from eodal.core.band import Band
from eodal.core.raster import RasterCollection
from eodal.core.sensors import Sentinel2
from eodal.operational.mapping.sentinel2 import Sentinel2Mapper
from eodal.utils.sentinel2 import ProcessingLevels, get_S2_platform_from_safe
from eodal.operational.mapping import MapperConfigs
from pathlib import Path
from typing import Dict, List, Optional, Union

from rtm_inv.core.config import RTMConfig, LookupTableBasedInversion
from rtm_inv.core.inversion import inv_img, retrieve_traits
from rtm_inv.core.lookup_table import generate_lut

logger = get_settings().logger
warnings.filterwarnings('ignore')

def traits_from_s2(
        date_start: date,
        date_end: date,
        aoi: Union[Path, gpd.GeoDataFrame],
        rtm_config: RTMConfig,
        output_dir: Path,
        lut_dir: Path,
        scene_cloud_cover_threshold: Optional[int] = 80,
        spatial_resolution: Optional[float] = 10.,
        resampling_method: Optional[int] = cv2.INTER_NEAREST_EXACT,
        processing_level: ProcessingLevels = ProcessingLevels.L2A,
        unique_feature_id: Optional[str] = None,
        **mapper_configs
    ) -> None: 
    """
    Retrieve traits from Sentinel-2 imagery using radiative transfer model
    inversion.

    The function works on a user-defined spatio-temporal subset of
    Sentinel-2 scenes and can provide 1:n traits, where n depends
    on the number of biophysical and -chemical parameters the RTM
    has available.

    :param date_start:
        start date for performing the trait retrieval
    :param date_end:
        end date for performing the trait retrieval
    :param aoi:
        area of interest for which to perform the trait retrieval
    :param rtm_config:
        radiative transfer model configuration (forward and inverse)
    :param output_dir:
        directory where to save the trait results (geoTiffs) to
    :param lut_dir:
        directory where to save lookup-tables to in order to speed
        up calculations (LUTs might be reused several times)
    :param scene_cloud_cover_threshold:
        maximum scene-wide cloud cover threshold to use in % (0-100). 
        f the `processing_level` is set to L2A, clouds, shadows
        and snow are masked based on the scene-classification layer.
        Default is 80%.
    :param spatial_resolution:
        spatial resolution of the output traits in meters. Default
        is 10 (m).
    :param resampling_method:
        resampling method to use when changing the spatial resolution
        of the bands. Default is `cv2.INTER_NEAREST_EXACT`.
    :param processing_level:
        Processing level of the Sentinel-2 data to use. Default is
        L2A (top-of-canopy). Depending on the RTM also L1C (top-of-
        atmosphere) is possible.
    :param unique_feature_id:
        optional attribute column in the `aoi` entry to be used as unique
        ID. If not available a random UUID is generated.
    :param mapper_configs:
        optional further mapping configurations to pass to
        `eodal.operational.mapping.MapperConfigs`
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

            # call function to generate lookup-table but first check if
            # does not exist already
            fname_lut = metadata.product_uri.values[0].replace('.SAFE', '_lut.pkl')
            fpath_lut = lut_dir.joinpath(fname_lut)
            if not fpath_lut.exists():
                # LUT does not exist; calculate it and save to pickle
                lut = generate_lut(
                    sensor=platform,
                    lut_params=rtm_config.rtm_params,
                    solar_zenith_angle=solar_zenith_angle,
                    viewing_zenith_angle=viewing_zenith_angle,
                    solar_azimuth_angle=solar_azimuth_angle,
                    viewing_azimuth_angle=viewing_azimuth_angle,
                    lut_size=rtm_config.lut_size,
                    sampling_method=rtm_config.method
                )
                lut.to_pickle(fpath_lut)
            else:
                # LUT exists, read it from pickle
                lut = pd.read_pickle(fpath_lut)

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
                cost_function=rtm_config.cost_function,
                n_solutions=rtm_config.n_solutions,
            )
            trait_img = retrieve_traits(
                lut=lut,
                lut_idxs=lut_idxs,
                traits=rtm_config.traits
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
            # calculate the NDVI in addition
            # feature_scene.calc_si('NDVI', inplace=True)
            # fname = f'{product_uri}_NDVI.tiff'
            # feature_scene.to_rasterio(
            #     fpath_raster=output_dir_feature.joinpath(fname),
            #     band_selection=['NDVI']
            # )
            logger.info(f'Feature {feature_id}: Finished inversion of {metadata.product_uri.iloc[0]}')

if __name__ == '__main__':

    def get_farms(data_dir: Path, farms: List[str], year: int) -> Dict[str, gpd.GeoDataFrame]:
        """
        Geometries of parcels for farms where LAI was collected
        """
        # loop over farms combine parcel geometries
        res = {}
        for farm in farms:
            parcels = []
            for fpath_farm_shp in data_dir.rglob(f'{farm}/WW_{year}/*.shp'):
                farm_shp = gpd.read_file(fpath_farm_shp)
                parcels.append(farm_shp[['geometry']].copy())
            farm_gdf = pd.concat(parcels)
            # dissolve geometries to get a single (Multi)Polygon per farm
            farm_gdf = farm_gdf.dissolve()
            farm_gdf['farm'] = farm
            res[farm] = farm_gdf
        return res

    data_dir = Path('/home/graflu/public/Evaluation/Projects/KP0031_lgraf_PhenomEn/02_Field-Campaigns')
    year = 2022
    farms = ['Strickhof'] # Witzwil

    # S2 configuration
    # maximum scene-cloud cover
    scene_cloud_cover_threshold = 50.

    # define start and end of the time series
    date_start = date(2022,2,1)
    date_end = date(2022,7,1)

    # spatial resolution of output product and spatial resampling method
    spatial_resolution = 10. # meters
    resampling_method = cv2.INTER_NEAREST_EXACT

    # set RTM parameters and traits to retrieve
    traits = ['lai', 'cab']
    rtm_params = Path('../parameters/prosail_frs.csv')
    method = 'FRS'
    # RTM configurations to test
    n_solutions = [1, 10, 100, 1000, 0.05, 0.1, 0.2]
    is_percentage = [False, False, False, False, True, True, True]
    cost_functions = ['rmse', 'squared_sum_of_differences', 'contrast_function']
    lut_sizes = [25000, 50000, 75000, 100000, 125000]

    # get all possible combinations
    combinations = list(itertools.product(*[n_solutions, cost_functions, lut_sizes]))

    # output directory for writing trait images
    output_dir = Path(
        '/home/graflu/public/Evaluation/Projects/KP0031_lgraf_PhenomEn/02_Field-Campaigns/Satellite_Data/FRS'
    )
    output_dir.mkdir(exist_ok=True)

    # save lookup-tables so they do no have to be recalculated
    lut_dir = output_dir.joinpath('lookup_tables')
    lut_dir.mkdir(exist_ok=True)

    # loop over farms and test inversion settings
    for farm_name in farms:
        # get field parcel geometries organized by farm
        farm_gdf_dict = get_farms(data_dir, [farm_name], year)
        aoi = farm_gdf_dict[farms[0]]
        aoi['name'] = farm_name

        # lookup-tables are stored per farm
        lut_dir_farm = lut_dir.joinpath(farm_name)
        lut_dir_farm.mkdir(exist_ok=True)

        # loop over combinations and run the inversion for each of them
        for combination in combinations:
    
            n_solution = combination[0]
            cost_function = combination[1]
            lut_size = combination[2]

            n_solution_idx = n_solutions.index(n_solution)
            n_solution_is_percentage = is_percentage[n_solution_idx]

            if n_solution_is_percentage:
                n_solutions_str = str(int(n_solution * 100))
                percentage_str = '%'
            else:
                n_solutions_str = str(int(n_solution))
                percentage_str = ''

            logger.info(
                f'Current Setup: Cost Function = {cost_function}; LUT Size = {lut_size}; ' + \
                f'Number of solutions: {n_solutions_str}{percentage_str}'
            )
    
            # output directory naming convention: <cost_function>_<lut-size>_<number-of-solutions>
            output_dir_combination = output_dir.joinpath(
                f'{cost_function}_{lut_size}_{n_solutions_str}{percentage_str}'
            )
            output_dir_combination.mkdir(exist_ok=True)

            # LUT directory for the given LUT size
            lut_dir_farm_and_size = lut_dir_farm.joinpath(str(lut_size))
            lut_dir_farm_and_size.mkdir(exist_ok=True)
    
            lut_config = LookupTableBasedInversion(
                traits=traits,
                n_solutions=n_solution,
                cost_function=cost_function,
                lut_size=lut_size,
                rtm_params=rtm_params,
                method=method
            )
        
            traits_from_s2(
                date_start=date_start,
                date_end=date_end,
                aoi=aoi,
                rtm_config=lut_config,
                scene_cloud_cover_threshold=scene_cloud_cover_threshold,
                spatial_resolution=spatial_resolution,
                resampling_method=resampling_method,
                unique_feature_id='name',
                output_dir=output_dir_combination,
                lut_dir=lut_dir_farm_and_size
            )
    
            logger.info(
                f'Finished Setup: Cost Function = {cost_function}; LUT Size = {lut_size}; Number of solutions: {n_solutions_str}%'
            )
