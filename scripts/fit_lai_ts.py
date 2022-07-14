'''
Fitting of

    - sigmoid (a.k.a. logistic) function
    - p-splines (monotonic increasing splines)

to S2 LAI data for the ascending ('green-up') branch of the LAI time series.
'''

import imageio
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from datetime import datetime
from eodal.config import get_settings
from eodal.core.band import Band, GeoInfo
from eodal.core.raster import RasterCollection
from pathlib import Path
from scipy.optimize import curve_fit

plt.style.use('ggplot')
logger = get_settings().logger

def sigmoid(x: np.ndarray | float, L: float, x0: float, k: float, b: float) -> np.ndarray | float:
    """
    Sigmoid (logistic) function

    :param x:
        data to fit. Can be a single value or ``np.ndarray``.
    :param L:
        scaling factor for scaling output from [0,1] to [0,L]
    :param x0:
        mid-point of the sigmoid where it should output 0.5
    :param k:
        scaling factor for the input [-inf, +inf]
    :param b:
        offset (bias) to change the output range from [0,L] to [b, L+b]
    :returns:
        value(s) of the sigmoid function for input ``x``
    """
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)

def fit_pspline(ydata: np.ndarray, max_iter: int = 30) -> np.ndarray:
    """
    found at: https://stats.stackexchange.com/questions/467126/monotonic-splines-in-python

    :param ydata:
        data to fit the curve to
    :returns:
        fitted data
    """
    # Prepare bases (Imat) and penalty
    N = ydata.size
    dd = 3
    E  = np.eye(N)
    D3 = np.diff(E, n=dd, axis=0)
    D1 = np.diff(E, n=1, axis=0)
    la = 100
    kp = 10000000
    # prepare ydata (ignore values after the maximum)
    _ydata = ydata.copy()
    # replace the values after the maximum LAI with this value to
    # improve the fitting along the ascending branch
    _ydata[np.nanargmax(_ydata)::] = np.nanmax(_ydata)
    # Monotone smoothing
    ws = np.zeros(N - 1)
    for it in range(max_iter):
        Ws      = np.diag(ws * kp)
        mon_cof = np.linalg.solve(E + la * D3.T @ D3 + D1.T @ Ws @ D1, _ydata)
        ws_new  = (D1 @ mon_cof < 0.0) * 1
        dw      = np.sum(ws != ws_new)
        ws      = ws_new
        if (dw == 0): break  

    return mon_cof

def fit_sigmoid(ydata: np.ndarray):
    """
    Fits a sigmoid (logistic) function to data points. Nodata values
    at the beginning of the time series are ignored. To force the
    logistic curve to reach maximum values, values after the maximum
    are set to the maximum value.

    :param ydata:
        data to fit the curve to
    :returns:
        fitted values
    """
    # this is an mandatory initial guess
    xdata = np.linspace(0, ydata.shape[0], num=ydata.shape[0], dtype=int)
    # check for nodata (0) - if all values are 0 do not fit anything
    if (ydata == 0).all():
        return ydata
    # fit sigmoid function to data points
    _ydata = ydata.copy()
    # set no-data values at the beginning of the time series to nan
    _ydata[_ydata == 0] = np.nan
    # replace the values after the maximum LAI with this value to
    # improve the fitting along the ascending branch
    _ydata[np.nanargmax(_ydata)::] = np.nanmax(_ydata)
    p0 = [np.nanmax(_ydata), np.median(xdata), 1, np.nanmin(_ydata)]
    try:
        popt, _ = curve_fit(
            sigmoid,
            xdata[~np.isnan(_ydata)],
            _ydata[~np.isnan(_ydata)],
            p0,
            method='lm'
        )
    except RuntimeError:
        return ydata
    y = sigmoid(xdata[~np.isnan(_ydata)], *popt)
    y2 = np.zeros_like(ydata)
    y2[np.isnan(_ydata)] = np.nan
    y2[~np.isnan(_ydata)] = y
    return y2

def plot_lai(ds_m: xr.DataArray, lai_raster: RasterCollection, method: str, out_dir: Path) -> None:
    """
    Animated (GIF) plotting of spatio-temporal dynamics of LAI.

    :param ds_m:
        reconstructed LAI values with masked no-data values
    :param lai_raster:
        corresponding ``eodal.core.raster.RasterCollection``
    :param method:
        method used for file-naming
    :param out_dir:
        directory where to save results (GIF files) to
    """
    fig_list = []
    max_lai = float(ds_m.lai.max())
    dates, medians, q05s, q95s = [], [], [], []
    date_start, date_end = lai_raster.band_names[0], lai_raster.band_names[-1]
    date_start = datetime.strptime(date_start, '%Y-%m-%d').date()
    date_end = datetime.strptime(date_end, '%Y-%m-%d').date()

    for date in lai_raster.band_names:
        f, axes = plt.subplots(nrows=2, ncols=1, figsize=(10,12))
        # axes = [axes]
        lai_sel = ds_m.sel(time=date)
        median = np.nanmedian(lai_sel.lai.values)
        q05 = np.nanquantile(lai_sel.lai.values, 0.05)
        q95 = np.nanquantile(lai_sel.lai.values, 0.95)
        lai_raster.plot_band(
            date,
            ax=axes[0],
            vmin=0,
            vmax=max_lai,
            colormap='viridis',
            colorbar_label=r'Green Leaf Area Index [$m2$/$m2$]'
        )
        axes[0].grid(False)
        dates.append(datetime.strptime(date, '%Y-%m-%d').date())
        medians.append(median)
        q05s.append(q05)
        q95s.append(q95)
        axes[1].plot(dates, medians, '-x', label='Median', color='blue')
        axes[1].fill_between(dates, q05s, q95s, label='Central 90%', color='orange', alpha=.5)
        axes[1].set_ylim(0, max_lai)
        axes[1].set_xlim(date_start, date_end)
        axes[1].set_ylabel(r'S2 GLAI [$m^2$/$m^2$]')
        axes[1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.35),
                       fancybox=False, shadow=False, ncol=2)
        plt.setp(axes[1].get_xticklabels(), rotation=90)
        f.canvas.draw()       # draw the canvas, cache the renderer
        image = np.frombuffer(f.canvas.tostring_rgb(), dtype='uint8')
        fig_list.append(image.reshape(f.canvas.get_width_height()[::-1] + (3,)))
        plt.close(f)

    # generate giff
    out_file = out_dir.joinpath(f'S2_lai_{method}_{aggregation}.gif') 
    imageio.mimsave(out_file, fig_list, fps=5)

def save_reconstructed_lai(ds_fit: xr.DataArray, out_dir: Path, method: str,
                           aggregation: str, plot: bool = True) -> None:
    """
    Write reconstructed LAI values to file and plot them (optional).

    :param ds_fit:
        ``xarray.DataArray`` with reconstructed values along time dimension
    :param out_dir:
        directory where to save results to (files + plots)
    :param method:
        fitting method used (for file-naming)
    :param aggregation:
        temporal aggregation used (e.g., '1D' for daily values)
    :param plot:
        if True (Default) plots animated time series (gif)
    """
    ds_m = ds_fit.where(ds_fit > 0)
    _dates = ds_m.time.values
    # save to file
    lai_values = ds_m.lai.values[0,:,:,:]

    # somehow the values are rotated by 90 degrees in the case of Arenenberg
    if out_dir.name == 'Arenenberg':
        # ds_m.lai.sel(time='2022-02-05').plot() # correct
        for idx in range(lai_values.shape[-1]):
            lai_values[:,:,idx] = np.rot90(np.rot90(np.rot90(ds_m.lai.values[0,:,:,idx]))).T
    # plt.imshow(lai_values[:,:,0])
    
    geo_info = GeoInfo(
        epsg=32632,
        ulx=float(ds_m.x.min()),
        uly=float(ds_m.y.max()),
        pixres_x=10,
        pixres_y=-10
    )
    lai_raster = RasterCollection()
    for idx, date in enumerate(_dates):
        lai_raster.add_band(
            band_constructor=Band,
            band_name=str(date)[0:10],
            values=lai_values[:,:,idx],
            geo_info=geo_info,
            unit='m2/m2'
        )
    fname = out_dir.joinpath(f'S2_lai_{method}_{aggregation}.tiff')
    lai_raster.to_rasterio(fname)
    # optionally plot the LAI spatio-temporal dynamics
    if plot:
        plot_lai(ds_m=ds_m, lai_raster=lai_raster, method=method, out_dir=out_dir)

def reconstruct_lai_ts(lai_dir: Path, out_dir: Path, aggregation: str, search_expr: str = '*_lai.tiff'):
    """
    """
    lai_list = []
    for lai_file in lai_dir.glob(search_expr):
        # read LAI from file and save it as xarray
        lai = RasterCollection.from_multi_band_raster(fpath_raster=lai_file)
        xr_lai = lai.to_xarray()
        # add the date
        sensing_date = datetime.strptime(lai_file.name.split('_')[2][0:8], '%Y%m%d')
        xr_lai = xr_lai.expand_dims(time=[sensing_date])
        # append to list
        lai_list.append(xr_lai)
    # concatenate LAI readings into a single xarray
    da = xr.concat(lai_list, dim='time')
    da = da.sortby('time')
    # interpolate nans linearly
    da = da.interpolate_na(dim='time', method='linear')
    # interpolate to daily values
    da = da.resample(time=aggregation).interpolate('linear')
    # fit the sigmoid function
    ds = da.to_dataset(name='lai')
    # set NaNs to zero
    ds = ds.fillna(0)

    ### curve fitting to obtain gap-free time series
    # fit p-spline (monotonically increasing spline)
    ds_spline = xr.apply_ufunc(
        fit_pspline,
        ds,
        input_core_dims=[['time']],
        output_core_dims=[['time']], 
        vectorize=True, 
        dask='parallelized', 
        output_dtypes=[np.float32]
    )
    save_reconstructed_lai(ds_fit=ds_spline, out_dir=out_dir, method='p-spline', aggregation=aggregation)

    # fit logistic curve
    ds_logistic = xr.apply_ufunc(
        fit_sigmoid,
        ds,
        input_core_dims=[['time']],
        output_core_dims=[['time']], 
        vectorize=True, 
        dask='parallelized', 
        output_dtypes=[np.float32]
    )
    save_reconstructed_lai(ds_fit=ds_logistic, out_dir=out_dir, method='logistic', aggregation=aggregation)

if __name__ == '__main__':
    farms = ['Arenenberg', 'Strickhof', 'Witzwil', 'SwissFutureFarm']
    aggregations = ['1D', '2D', '3D', '4D', '5D', '10D']
    for farm in farms:
        logger.info(f'Working on {farm}')
        lai_dir = Path(
            f'/home/graflu/public/Evaluation/Projects/KP0031_lgraf_PhenomEn/02_Field-Campaigns/Satellite_Data/{farm}'
        )
        out_dir = lai_dir
        for aggregation in aggregations:
            logger.info(f'Farm: {farm} - temporal aggregation: {aggregation} - Started')
            reconstruct_lai_ts(
                lai_dir=lai_dir,
                out_dir=out_dir,
                aggregation=aggregation,
                search_expr='*_lai.tiff'
            )
            logger.info(f'Farm: {farm} - temporal aggregation: {aggregation} - Finished')
