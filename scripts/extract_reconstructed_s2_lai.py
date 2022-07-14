'''
Created on Jul 13, 2022

@author: graflu
'''

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error

from eodal.core.raster import RasterCollection
from pathlib import Path
from typing import List

plt.style.use('ggplot')

def calc_rmse(y_actual: np.ndarray, y_predicted: np.ndarray) -> float:
    """returns the root mean squared error (RMSE)"""
    return mean_squared_error(y_actual, y_predicted, squared=False)

def extract_data(lai_dir: Path, methods: List[str], fpath_insitu: Path, farm: str,
                 aggregation: str) -> gpd.GeoDataFrame:
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
        fpath_fitted = lai_dir.joinpath(f'S2_lai_{method}_{aggregation}.tiff')
        fitted = RasterCollection.from_multi_band_raster(fpath_raster=fpath_fitted)
        res_list = []
        for date, gdf_farm_date in gdf_farm:
            try:
                res = fitted.get_pixels(
                    vector_features=gdf_farm_date,
                    band_selection=[str(date)]
                )
                res = res.rename(columns={str(date): f'S2 LAI {method}'})
                res_list.append(res)
            except KeyError:
                # when the temporal resolution is not 1d, there might be no match
                continue
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

def plot_scatter(lai_df: gpd.GeoDataFrame, out_dir: Path):
    """
    """
    methods = ['logistic', 'p-spline']
    f, axes = plt.subplots(nrows=1, ncols=len(methods), figsize=(20,10))
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # exclude unplausible high LAI readings
    lai_df = lai_df[lai_df.lai <= 7]
    for idx, method in enumerate(methods):
        sns.scatterplot(
            x='lai',
            y=f'S2 LAI {method}',
            data=lai_df,
            ax=axes[idx],
            hue='location'
        )
        axes[idx].plot([x for x in range(8)], [x for x in range(8)], linestyle='dashed', label='1:1 line')
        axes[idx].set_xlim(0, 7)
        axes[idx].set_ylim(0, 7)
        axes[idx].set_xlabel(r'In-Situ Green Leaf Area Index [$m^2$/$m^2$]')
        axes[idx].set_ylabel(r'Sentinel-2 Green Leaf Area Index [$m^2$/$m^2$]')
        axes[idx].set_title(f'{method} - Temporal resolution: {aggregation}')
        # goodness of fit
        N = lai_df.shape[0]
        rmse = calc_rmse(y_actual=lai_df.lai.values, y_predicted=lai_df[f'S2 LAI {method}'].values)
        nrmse = rmse / lai_df.lai.values.mean() * 100.
        median_error = (lai_df.lai - lai_df[f'S2 LAI {method}']).median()
        nmad = 1.4826 * (abs(lai_df.lai - lai_df[f'S2 LAI {method}'])).median()
        slope, intercept, r_value, p_value, std_err = linregress(lai_df.lai.values, lai_df[f'S2 LAI {method}'].values)
        textstr = f'N = {N}\nRMSE = {np.round(rmse,2)} ' + r'$m^2$/$m^2$' + f'\nnRMSE = {np.round(nrmse,2)} %' + \
            f'\nNMAD = {np.round(nmad,2)} ' + r'$m^2$/$m^2$' + f'\nPearson`s R = {np.round(r_value,2)}'
        # place a text box in upper left in axes coordinates
        axes[idx].text(0.65, 0.2, textstr, transform=axes[idx].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    fname = out_dir.joinpath(f'all_lai_validation_scatter_{aggregation}.png')
    f.savefig(fname, bbox_inches='tight')
    

if __name__ == '__main__':

    methods = ['logistic', 'p-spline']
    farms = ['Arenenberg', 'Strickhof', 'Witzwil', 'SwissFutureFarm']
    aggregations = ['1D', '2D', '3D', '4D', '5D', '10D']

    fpath_insitu = Path(
        '/home/graflu/public/Evaluation/Projects/KP0031_lgraf_PhenomEn/02_Field-Campaigns/in-situ_traits_2022/in-situ_glai_all_measurements.gpkg'
    )

    for aggregation in aggregations:
        farm_dfs = []
        for farm in farms:
            lai_dir = Path(
                f'/home/graflu/public/Evaluation/Projects/KP0031_lgraf_PhenomEn/02_Field-Campaigns/Satellite_Data/{farm}'
                )
            farm_df = extract_data(lai_dir, methods, fpath_insitu, farm, aggregation)
            farm_dfs.append(farm_df)
        large_df = pd.concat(farm_dfs)

        out_dir = Path(
            f'/home/graflu/public/Evaluation/Projects/KP0031_lgraf_PhenomEn/02_Field-Campaigns/in-situ_traits_2022/green_leaf_area_index/reconstructed_{aggregation}'
        )
        out_dir.mkdir(exist_ok=True)
        plot_scatter(lai_df=large_df, out_dir=out_dir)


