'''
Created on Jul 13, 2022

@author: graflu
'''

import geopandas as gpd
import pandas as pd

from eodal.core.raster import RasterCollection
from pathlib import Path
from typing import List

def extract_data(lai_dir: Path, methods: List[str], fpath_insitu: Path, farm: str
                 ) -> gpd.GeoDataFrame:
    """
    """
    # open in-situ data
    gdf = gpd.read_file(fpath_insitu)
    # shrink to farm
    gdf_farm = gdf[gdf.location == farm].copy()
    gdf_farm['date'] = gdf_farm['date'].apply(lambda x: x.replace('T', ' '))
    gdf_farm = gdf_farm[~gdf_farm.date.isin(['Na ', 'None'])]
    gdf_farm.date = pd.to_datetime(gdf_farm.date).dt.date
    # loop through in-situ dates and extract the fitted LAI values from the
    # different methods
    gdf_farm = gdf_farm.groupby('date')
    method_res = []
    for method in methods:
        fpath_fitted = lai_dir.joinpath(f'S2_lai_{method}.tiff')
        fitted = RasterCollection.from_multi_band_raster(fpath_raster=fpath_fitted)
        res_list = []
        for date, gdf_farm_date in gdf_farm:
            res = fitted.get_pixels(
                vector_features=gdf_farm_date,
                band_selection=[str(date)]
            )
            res = res.rename(columns={str(date): f'S2 LAI {method}'})
            res_list.append(res)
        method_df = pd.concat(res_list)
        method_res.append(method_df)
    # combine all results
    master = method_res[0]
    slaves = method_res[1::]
    for slave in slaves:
        master = master.join(slave, rsuffix='r_', lsuffix='l_')
        master = master.rename(columns={'datel_': 'date', 'locationl_': 'location', 'lail_': 'lai'})
        drop_cols = [x for x in master.columns if x.endswith('r_') or x.endswith('l_')]
        master.drop(drop_cols, axis=1, inplace=True)
    return master

def plot_scatter(lai_df: gpd.GeoDataFrame):
    pass

if __name__ == '__main__':

    methods = ['logistic', 'p-spline']
    farms = ['Arenenberg', 'Strickhof', 'Witzwil', 'SwissFutureFarm']

    fpath_insitu = Path(
        '/home/graflu/public/Evaluation/Projects/KP0031_lgraf_PhenomEn/02_Field-Campaigns/in-situ_traits_2022/in-situ_glai_all_measurements.gpkg'
    )

    farm_dfs = []
    for farm in farms:
        lai_dir = Path(
            f'/home/graflu/public/Evaluation/Projects/KP0031_lgraf_PhenomEn/02_Field-Campaigns/Satellite_Data/{farm}'
        )
        farm_df = extract_data(lai_dir, methods, fpath_insitu, farm)
        farm_dfs.append(farm_df)

