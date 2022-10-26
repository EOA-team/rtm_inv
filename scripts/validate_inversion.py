'''
Created on Jul 7, 2022

@author: graflu
'''

import geopandas as gpd
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from eodal.core.raster import RasterCollection
from pathlib import Path
from scipy.stats import linregress, kstest
from sklearn.metrics import mean_squared_error

plt.style.use('ggplot')

def calc_rmse(y_actual: np.ndarray, y_predicted: np.ndarray) -> float:
    """returns the root mean squared error (RMSE)"""
    return mean_squared_error(y_actual, y_predicted, squared=False)

def scatter(merged: gpd.GeoDataFrame, farm: str, out_dir: Path):
    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    sns.scatterplot(x='lai', y='s2_lai', data=merged, ax=ax, hue='protocol_version')
    ax.plot([x for x in range(0,8)], [x for x in range(0,8)],
            linestyle='dashed', color='blue', label='1:1 line')
    ax.set_ylim(0,7)
    ax.set_xlim(0,7)
    ax.set_title(
        f'{farm} 2022 (N={merged.shape[0]})\n' + r'$\Delta t_{max}$ = ' + str(merged.time_delta.max())
    )
    ax.set_ylabel(r'Sentinel-2 Green Leaf Area Index [$m^2$/$m^2$]')
    ax.set_xlabel(r'In-situ Green Leaf Area Index [$m^2$/$m^2$]')
    # goodness of fit
    N = merged.shape[0]
    rmse = calc_rmse(y_actual=merged.lai.values, y_predicted=merged.s2_lai.values)
    nrmse = rmse / merged.lai.values.mean() * 100.
    median_error = (merged.lai - merged.s2_lai).median()
    nmad = 1.4826 * abs(median_error)
    # linear regression
    slope, intercept, r_value, p_value, std_err = linregress(
        merged.lai.values, merged.s2_lai.values
    )
    # check if the residues follow a Gaussian distribution
    modelled = slope * np.linspace(0,8, num=merged.s2_lai.values.shape[0]) + intercept
    ax.plot(np.linspace(0,8, num=merged.s2_lai.values.shape[0]), modelled,
            label=f'Linear Regression (p={np.round(p_value,3)})')
    residues = merged.s2_lai.values - modelled
    stats, p_value = kstest(residues, 'norm')
    # residues are Gaussian if p > 0.05
    residues_gaussian = 'Probably Not Gaussian'
    if p_value > 0.05:
        residues_gaussian = 'Probably Gaussian'
    textstr = f'N = {N}\nRMSE = {np.round(rmse,2)} ' + r'$m^2$/$m^2$' + f'\nnRMSE = {np.round(nrmse,2)} %' + \
        f'\nNMAD = {np.round(nmad,2)} ' + r'$m^2$/$m^2$' + f'\nPearson`s R = {np.round(r_value,2)}' + \
        f'\nResidues: {residues_gaussian}'
    # save statistics
    stats = {
        'N': N,
        'RMSE': rmse,
        'nRMSE': nrmse,
        'median error': median_error,
        'NMAD': nmad,
        'Lin-Regress Pearson`s R': r_value,
        'Lin-Regress P': p_value,
        'Lin-Regress Slope': slope,
        'Residues': residues_gaussian
    }
    fname_stats = out_dir.joinpath(f'{farm}_error_stats.json')
    with open(fname_stats, 'w+') as dst:
        json.dump(stats, dst)

    # place a text box in upper left in axes coordinates
    ax.text(0.55, 0.2, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    ax.legend()
    fname = out_dir.joinpath(f'{farm}_lai_validation_scatter.png')
    f.savefig(fname, bbox_inches='tight')
    plt.close(f)

def ts(merged: gpd.GeoDataFrame, farm: str, out_dir: Path, location: str = None):
    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,7))
    if location is None:
        sns.scatterplot(x='date', y='lai', data=merged, marker='o', label='in-situ', ax=ax)
        sns.scatterplot(x='time', y='s2_lai', data=merged, marker='x', label='Sentinel-2', ax=ax)
    else:
        sns.scatterplot(x='date', y='lai', data=merged, marker='o', label='in-situ', ax=ax, hue=location)
        sns.scatterplot(x='time', y='s2_lai', data=merged, marker='x', label='Sentinel-2', ax=ax, hue=location)
    ax.set_ylim(0,7)
    ax.set_ylabel(r'Green Leaf Area Index [$m^2$/$m^2$]')
    ax.set_title(
        f'{farm} 2022 (N={merged.shape[0]})\n' + r'$\Delta t_{max}$ = ' + str(merged.time_delta.max())
    )
    fname = out_dir.joinpath(f'{farm}_lai_validation_ts.png')
    f.savefig(fname, bbox_inches='tight')
    plt.close(f)

def validate_glai(fpath_in_situ: Path, fpath_sat_data: Path, out_dir: Path) -> None:
    """
    """
    in_situ = gpd.read_file(fpath_in_situ)
    # get single farm locations
    farms = in_situ.location.unique()
    all_dfs = []
    # loop over farms and extract satellite data matching in-situ dates
    for farm in farms:
        farm_insitu = in_situ[in_situ.location == farm].copy()
        # drop un-realistically high LAI values
        farm_insitu = farm_insitu[farm_insitu['lai'] <= 8]
        farm_insitu.date = pd.to_datetime(farm_insitu.date)
        farm_insitu.sort_values(by='date', inplace=True)
        # get S2 traits and dates
        sat_dir = fpath_sat_data.joinpath(farm)
        if not sat_dir.exists(): continue
        lai_files = [x for x in sat_dir.glob('*_traits.tiff')]
        lai_files_df = pd.DataFrame({'file': lai_files})
        lai_files_df['time'] = lai_files_df.file.apply(
            lambda x: x.name.split('_')[2]
        )
        lai_files_df.time = pd.to_datetime(lai_files_df.time)
        lai_files_df.sort_values(by='time', inplace=True)

        # find closest temporal matches between in-situ and files with S2 observations
        merged = pd.merge_asof(farm_insitu, lai_files_df, left_on="date", right_on="time", direction='nearest')
        merged['time_delta'] = merged['date'] - merged['time']
        # filter out records where the time delta between in-situ and S2 is too high
        merged = merged[merged.time_delta <= pd.Timedelta(2, unit='d')].copy()
        # extract S2 LAI values from files
        merged['s2_lai'] = merged.apply(
            lambda x, merged=merged, RasterCollection=RasterCollection:
                RasterCollection.read_pixels(
                    fpath_raster=x['file'],
                    vector_features=gpd.GeoDataFrame(geometry=[x.geometry], crs=merged.crs)
                )['lai'].values[0],
            axis=1 
        )
        merged = merged[~np.isnan(merged.s2_lai)]
        scatter(merged, farm, out_dir=out_dir)
        ts(merged, farm, out_dir=out_dir)
        all_dfs.append(merged.copy())

    gdf = pd.concat(all_dfs)
    fname = out_dir.joinpath('s2_in-situ_green_leaf_area_index.gpkg')
    gdf['file'] = gdf['file'].apply(lambda x: str(x))
    gdf['time_delta'] = gdf['time_delta'].apply(lambda x: str(x))
    gdf.to_file(fname, driver='GPKG')
    scatter(gdf, 'all', out_dir=out_dir)
    ts(gdf, 'all', location='location', out_dir=out_dir)

def rank_model_runs(result_dir: Path, output_dir: Path):
    # loop over the result sub directories assuming that there's nothing else
    res = []
    for subdir in result_dir.iterdir():
        splitted = subdir.name.split('_')
        n_solutions = splitted[-1]
        lut_size = splitted[-2]
        cost_function = splitted[:-2]
        cost_function = ' '.join(cost_function)
        # load the error statistics (all sites)
        json_fpath = subdir.joinpath('all_error_stats.json')
        if not json_fpath.exists(): continue
        with open(json_fpath, 'r') as fp:
            stats = json.load(fp)
        stats['cost_function'] = cost_function
        stats['lut_size'] = lut_size
        stats['n_solutions_perc'] = n_solutions
        res.append(stats)

    res_df = pd.DataFrame(res)
    fname = output_dir.joinpath('all_combined_error_stats.csv')
    res_df.to_csv(fname, index=False)     

if __name__ == '__main__':

    import itertools

    fpath_in_situ = Path(
        # '/mnt/ides/Lukas/04_Work/PhenomEn_GLAI/in_situ/Survey_points.gpkg'
        '/home/graflu/public/Evaluation/Projects/KP0031_lgraf_PhenomEn/02_Field-Campaigns/in-situ_traits_2022/in-situ_glai_all_measurements.gpkg'
    )
    fpath_sat_data = Path(
        # '/mnt/ides/Lukas/04_Work/PhenomEn_GLAI/Satellite_Data'
        '/home/graflu/public/Evaluation/Projects/KP0031_lgraf_PhenomEn/02_Field-Campaigns/Satellite_Data/LHS'
    )
    out_dir = Path(
        # '/mnt/ides/Lukas/04_Work/PhenomEn_GLAI/analysis'
        '/home/graflu/public/Evaluation/Projects/KP0031_lgraf_PhenomEn/02_Field-Campaigns/in-situ_traits_2022/green_leaf_area_index/raw_data/LHS'
    )
    out_dir.mkdir(exist_ok=True)

    # test different ways to perform the inversion (LUT size, cost function, number of solutions)
    n_solutions = [1, 10, 100, 1000, 0.05, 0.1, 0.2]
    cost_functions = ['rmse', 'squared_sum_of_differences', 'contrast_function']
    lut_sizes = [25000, 50000, 75000, 100000, 125000]
    is_percentage = [False, False, False, False, True, True, True]

    combinations = itertools.product(*[n_solutions, cost_functions, lut_sizes])

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

        combo_str = f'{cost_function}_{lut_size}_{n_solutions_str}{percentage_str}'
    
        out_dir_combo = out_dir.joinpath(combo_str)
        out_dir_combo.mkdir(exist_ok=True)
        fpath_sat_data_combo = fpath_sat_data.joinpath(combo_str)
    
        if not fpath_sat_data_combo.exists():
            continue
    
        try:
            validate_glai(fpath_in_situ, fpath_sat_data_combo, out_dir_combo)
        except Exception:
            continue

    # rank the model setup's in terms of different error metrics
    rank_model_runs(result_dir=out_dir, output_dir=out_dir)
