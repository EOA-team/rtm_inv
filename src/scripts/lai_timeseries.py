'''
Created on Feb 17, 2022

@author: graflu
'''

import pandas as pd
from pathlib import Path

from agrisatpy.core.raster import RasterCollection
from agrisatpy.core.band import Band

search_expr_lai ='S2*_MSIL2A_*.VIs'
# search GPR LAI and ProSAIL LAI
search_expr_lai_prosail = 'VI_*None_10m_GLAI.tif'
search_expr_lai_gpr = 'VI_*None_10m_LAI.tif'
# search NDVI (serves as reference)
search_expr_ndvi = 'VI_*None_10m_NDVI.tif'
# scene classification layer to mask out clouds and shadows
search_expr_scl = '*_SCL.tiff'

def extract_data(
        lai_dir: Path,
        parcels: Path,
        output_fname: Path
    ):
    """
    Extracts the NDVI, GPR-derived GLAI and ProSAIL-derived GLAI from a series
    of Sentinel-2 scenes. Uses the scene classification layer to mask out all
    pixels not classified as vegetation (4) or bare soil (5)
    """
    df_list = []
    for scene in lai_dir.rglob(str(search_expr_lai)):

        sensing_date = pd.to_datetime(scene.name.split('_')[2][0:8])

        vi_dir = scene.joinpath('Vegetation_Indices')

        ndvi_file = next(vi_dir.glob(search_expr_ndvi))
        prosail_lai_file = next(vi_dir.glob(search_expr_lai_prosail))
        gpr_lai_file = next(vi_dir.glob(search_expr_lai_gpr))

        scl_dir = scene.joinpath('scene_classification')
        scl_file = next(scl_dir.glob(search_expr_scl))

        # read data from raster into RasterCollection
        collection = RasterCollection()
        collection.add_band(
            band_constructor=Band.from_rasterio,
            fpath_raster=ndvi_file,
            band_idx=1,
            band_name_dst='NDVI',
            vector_features=parcels
        )
        collection.add_band(
            band_constructor=Band.from_rasterio,
            fpath_raster=prosail_lai_file,
            band_idx=1,
            band_name_dst='ProSAIL GLAI',
            vector_features=parcels
        )
        collection.add_band(
            band_constructor=Band.from_rasterio,
            fpath_raster=gpr_lai_file,
            band_idx=1,
            band_name_dst='GPR GLAI',
            vector_features=parcels
        )
        collection.add_band(
            band_constructor=Band.from_rasterio,
            fpath_raster=gpr_lai_file,
            band_idx=2,
            band_name_dst='GPR GLAI SD',
            vector_features=parcels
        )
        collection.add_band(
            band_constructor=Band.from_rasterio,
            fpath_raster=scl_file,
            band_idx=1,
            band_name_dst='SCL',
            vector_features=parcels
        )
        collection.add_band(
            band_constructor=Band.from_vector,
            vector_features=parcels,
            geo_info=collection['NDVI'].geo_info,
            band_name_src='crop_code',
            band_name_dst='crop_code',
            snap_bounds=collection['NDVI'].bounds
        )
        collection.mask(
            mask=collection['NDVI'].values.mask,
            bands_to_mask=['crop_code'],
            inplace=True
        )

        # convert to GeoDataFrame
        gdf = collection.to_dataframe()
        # keep SCL classes 4 and 5, only
        clean_pix_gdf = gdf[gdf['SCL'].isin([4,5])].copy()
        clean_pix_gdf['date'] = sensing_date
        clean_pix_gdf['date'] = pd.to_datetime(clean_pix_gdf['date'])
        df_list.append(clean_pix_gdf)

        print(f'Extracted scene {scene.name}')

    # concat dataframe, clean it and save it to csv
    complete_gdf = pd.concat(df_list)
    complete_gdf['x'] = complete_gdf.geometry.x
    complete_gdf['y'] = complete_gdf.geometry.y
    complete_gdf.drop('geometry', axis=1, inplace=True)
    complete_gdf.to_csv(output_fname, index=False)


if __name__ == '__main__':

    # set input and output paths
    parcels = Path('/mnt/ides/Lukas/software/scripts_paper_uncertainty/shp/ZH_Polygons_2019_EPSG32632_selected-crops_buffered.shp')
    lai_dir = Path('/home/graflu/Documents/uncertainty/S2_MSIL1C_orig')
    output_fname = Path('/home/graflu/Documents/uncertainty/NDVI_GPR-ProSAIL_GLAI_values_crops.csv')

    extract_data(lai_dir, parcels, output_fname)
    

    
    