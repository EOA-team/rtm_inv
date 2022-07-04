'''
Script to
* extract the S2 LAI data
* plot time series for selected sampling points where ground data is available
* compare S2 LAI to LAI estimates derived from PlanetScope
'''

import datetime
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eodal.core.raster import RasterCollection
from pathlib import Path
from sklearn.metrics import r2_score

plt.style.use('ggplot')

def extract_data(
        fpath_points: Path,
        s2_lai_dir: Path
    ) -> gpd.GeoDataFrame:
    
    # get sampling points
    points = gpd.read_file(fpath_points)
    point_geoms = gpd.GeoDataFrame(geometry=points.geometry.unique(), crs=points.crs)
    point_ids = points[['geometry', 'Point_ID_caller']].copy()
    point_ids = point_ids.drop_duplicates()
    
    # loop over S2 LAI files and extract the values from the S2 pixels closest to the sampling points
    s2_lai_res = []
    for s2_lai_file in s2_lai_dir.glob('*_traits.tiff'):
        
        s2_lai = RasterCollection.read_pixels(
            fpath_raster=s2_lai_file,
            vector_features=point_geoms
        )
        # check if there was data (could also be NaN because of clouds)
        if np.isnan(s2_lai.lai).all():
            continue
    
        # save the sensing data (from file name)
        s2_lai['acquired'] = datetime.datetime.strptime(
            s2_lai_file.name.split('_')[2], '%Y%m%dT%H%M%S'
        )
        # append to list - copy() is crucial here!!!
        s2_lai_res.append(s2_lai.copy())
    
    s2_lai = pd.concat(s2_lai_res)
    # join the point id from the input layer
    s2_lai = gpd.sjoin(s2_lai, point_ids, how='inner')
    return s2_lai

def plot_histogram(s2_lai: gpd.geoseries, out_dir: Path):
    # plot LAI histogram
    f, ax = plt.subplots(ncols=1, nrows=1)
    s2_lai.lai.hist(bins=50, ax=ax, density=False)
    ax.set_xlabel(r'Sentinel-2 Green Leaf Area Index [$m^2$/$m^2$]')
    ax.set_ylabel('Frequency [-]')
    f.savefig(out_dir.joinpath('all_s2_lai_histogram.png'), bbox_inches='tight')
    plt.close(f)

def plot_lai_ts(gdf: gpd.GeoDataFrame, out_dir: Path):
    # visualize LAI time series for the sampling points
    points = gdf.Point_ID_caller.unique()
    
    for point in points:
        try:
            point_pixels = gdf[gdf.Point_ID_caller == point].copy()
            point_pixels.acquired = pd.to_datetime(point_pixels.acquired)
            point_pixels['date'] = point_pixels.acquired.apply(lambda x: x.date())
            point_pixels = point_pixels.sort_values(by='acquired')
            f, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 7))
            ax.scatter(point_pixels.acquired, point_pixels.lai, linestyle='solid')
            ax.set_ylabel(r'Sentinel-2 Green Leaf Area Index [$m^2$/$m^2$]')
            ax.set_xlabel('Time')
            ax.set_title(f'Bramenwies - Point {point}')
            ax.set_ylim(0,7)
            plt.xticks(rotation=45)
            fname = out_dir.joinpath(f'{point}_ps_lai_timeseries.png')
            f.savefig(fname, bbox_inches='tight')
            plt.close(f)
        except Exception as e:
            print(e)

def lai_s2_vs_ps(s2_lai: gpd.GeoDataFrame, ps_lai: gpd.GeoDataFrame, out_dir: Path):
    # select those Planet LAI values were also S2 data is available
    ps_lai.acquired_other = pd.to_datetime(ps_lai.acquired_other)
    ps_lai['date'] = ps_lai.acquired_other.apply(lambda x: x.date())
    s2_lai['date'] = s2_lai.acquired.apply(lambda x: x.date())
    merged = pd.merge(
        s2_lai,
        ps_lai,
        left_on=['geometry', 'date'],
        right_on=['geometry', 'date'],
        suffixes=['_s2', '_ps']
    )
    # drop all records were no S2 data is available
    merged.dropna(inplace=True)
    # plot scatter plot
    f, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 7))
    ax.scatter(merged.lai_s2, merged.lai_ps)
    ax.plot([x for x in range(0,8)], [x for x in range(0,8)], linestyle='dashed', color='blue')
    ax.set_xlim(0,7)
    ax.set_ylim(0,7)
    ax.set_xlabel(r'Sentinel-2 Green Leaf Area Index [$m^2$/$m^2$]')
    ax.set_ylabel(r'PlanetScope Green Leaf Area Index [$m^2$/$m^2$]')
    r2 = r2_score(merged.lai_s2, merged.lai_ps)
    ax.annotate(r'$R^2$' + f' = {np.round(r2, 2)}', (5, 6), fontsize=14)
    fname = out_dir.joinpath('lai_scatter_all_pixels.png')
    f.savefig(fname, bbox_inches='tight')
    plt.close(f)

if __name__ == '__main__':

    fpath_points = Path('/mnt/ides/Lukas/04_Work/PS_Eschikon_TS/timeseries_BW_medians_lai_cleaned.gpkg')
    # s2_lai_dir = Path('/home/graflu/public/Evaluation/Hiwi/2022_samuel_wildhaber_MSc/S2_LAI/')
    s2_lai_dir = Path('/mnt/ides/Lukas/04_Work/S2_LAI_ProSAIL')
    # extract LAI data from S2 LAI files
    s2_lai = extract_data(fpath_points, s2_lai_dir)

    # plot data
    out_dir = s2_lai_dir.joinpath('figures')
    out_dir.mkdir(exist_ok=True)
    plot_histogram(s2_lai, out_dir)
    plot_lai_ts(gdf=s2_lai, out_dir=out_dir)

    # compare Planet and S2 LAI
    ps_lai = gpd.read_file(fpath_points)
    out_dir = s2_lai_dir.joinpath('S2_PS_LAI_Scatter')
    out_dir.mkdir(exist_ok=True)
    lai_s2_vs_ps(s2_lai=s2_lai, ps_lai=ps_lai, out_dir=out_dir)
    