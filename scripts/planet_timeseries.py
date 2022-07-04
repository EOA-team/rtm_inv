'''
Plots time series of Planet-Scope traits (LAI, NDVI, etc.)
'''

from pathlib import Path
from typing import Tuple

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')

def plot_ts(
        gdf: gpd.GeoDataFrame, column_time: str, column_point_id: str, trait: str,
        trait_lims: Tuple[float, float], trait_label: str,  out_dir: Path
    ) -> None:
    
    # plot LAI histogram
    f, ax = plt.subplots(ncols=1, nrows=1)
    gdf[trait].hist(bins=50, ax=ax, density=False)
    ax.set_xlabel(trait_label)
    ax.set_ylabel('Frequency [-]')
    ax.set_xlim(trait_lims)
    f.savefig(out_dir.joinpath(f'all_ps_{trait}_histogram.png'), bbox_inches='tight')
    
    # visualize LAI time series for the sampling points
    points = gdf[column_point_id].unique()
    
    for point in points:
        try:
            point_pixels = gdf[gdf[column_point_id] == point].copy()
            point_pixels[column_time] = pd.to_datetime(point_pixels[column_time])
            point_pixels['date'] = point_pixels[column_time].apply(lambda x: x.date())
            point_pixels = point_pixels.sort_values(by=column_time)
            f, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 7))
            ax.scatter(point_pixels[column_time], point_pixels[trait])
            ax.set_ylabel(trait_label)
            ax.set_xlabel('Time')
            ax.set_title(f'Bramenwies - Point {point}')
            ax.set_ylim(trait_lims)
            plt.xticks(rotation=45)
            fname = out_dir.joinpath(f'{point}_ps_{trait}_timeseries.png')
            f.savefig(fname, bbox_inches='tight')
            plt.close(f)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    fpath = Path('/mnt/ides/Lukas/04_Work/PS_Eschikon_TS/timeseries_BW_medians_lai_cleaned.gpkg')
    # fpath = Path('/mnt/ides/Lukas/04_Work/PS_Eschikon_TS/timeseries_BW_lai_cleaned.gpkg')
    gdf = gpd.read_file(fpath)
    
    out_dir = fpath.parent.joinpath('figures_medians')
    # out_dir = fpath.parent.joinpath('figures_all_pixels')
    out_dir.mkdir(exist_ok=True)
    
    column_time = 'acquired_other'
    # column_time = 'acquired'
    column_point_id = 'Point_ID_caller'
    # column_point_id = 'Point_ID'

    # LAI
    # trait_label = r'PlanetScope Green Leaf Area Index [$m^2$/$m^2$]'
    # trait_lims = (0,7)
    # trait = 'lai'

    # NDVI
    trait_label = r'PlanetScope NDVI [-]'
    trait_lims = (0,1)
    trait = 'NDVI'

    plot_ts(gdf, column_time, column_point_id, trait, trait_lims, trait_label, out_dir)
