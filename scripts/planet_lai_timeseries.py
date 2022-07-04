'''
Plots time series of Planet-Scope LAI
'''

from pathlib import Path

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')

def plot_ts(
        gdf: gpd.GeoDataFrame, column_time: str, column_point_id: str, out_dir: Path
    ) -> None:
    
    # plot LAI histogram
    f, ax = plt.subplots(ncols=1, nrows=1)
    gdf.lai.hist(bins=50, ax=ax, density=False)
    ax.set_xlabel(r'PlanetScope Green Leaf Area Index [$m^2$/$m^2$]')
    ax.set_ylabel('Frequency [-]')
    f.savefig(out_dir.joinpath('all_ps_lai_histogram.png'), bbox_inches='tight')
    
    # visualize LAI time series for the sampling points
    points = gdf[column_point_id].unique()
    
    for point in points:
        try:
            point_pixels = gdf[gdf[column_point_id] == point].copy()
            point_pixels[column_time] = pd.to_datetime(point_pixels[column_time])
            point_pixels['date'] = point_pixels[column_time].apply(lambda x: x.date())
            point_pixels = point_pixels.sort_values(by=column_time)
            f, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 7))
            ax.scatter(point_pixels[column_time], point_pixels.lai)
            ax.set_ylabel(r'PlanetScope Green Leaf Area Index [$m^2$/$m^2$]')
            ax.set_xlabel('Time')
            ax.set_title(f'Bramenwies - Point {point}')
            ax.set_ylim(0,7)
            plt.xticks(rotation=45)
            fname = out_dir.joinpath(f'{point}_ps_lai_timeseries.png')
            f.savefig(fname, bbox_inches='tight')
            plt.close(f)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    # fpath = Path('/mnt/ides/Lukas/04_Work/PS_Eschikon_TS/timeseries_BW_medians_lai_cleaned.gpkg')
    fpath = Path('/mnt/ides/Lukas/04_Work/PS_Eschikon_TS/timeseries_BW_lai_cleaned.gpkg')
    gdf = gpd.read_file(fpath)
    
    # out_dir = fpath.parent.joinpath('figures_medians')
    out_dir = fpath.parent.joinpath('figures_all_pixels')
    out_dir.mkdir(exist_ok=True)
    
    # column_time = 'acquired_other'
    column_time = 'acquired'
    # column_point_id = Point_ID_Caller
    column_point_id = 'Point_ID'

    plot_ts(gdf, column_time, column_point_id, out_dir)
