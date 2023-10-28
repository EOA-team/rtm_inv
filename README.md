# rtm_inv - Radiative Transfer Modelling and Inversion in Python

This repository allows you to run radiative transfer models (RTM) to model the optical properties of vegetation canopies (mainly crops).
`rtm_inv` is essentially a "backend" repository containing

* functions to generate lookup-tables from forward runs of [PROSAIL](http://teledetection.ipgp.jussieu.fr/prosail/) and (experimentally) [SPART](https://doi.org/10.1016/j.rse.2020.111870).
* functions to "invert" optical data by comparing observed with simulated spectra to obtain canopy and leaf traits from optical (satellite) imagery

The focus of `rtm_inv` currently is on optical satellite missions including [Sentinel2A and B](https://sentinel.esa.int/web/sentinel/missions/sentinel-2), [Landsat 8 and 9](https://landsat.gsfc.nasa.gov/satellites/landsat-9/), and [PlanetScope SuperDove](https://pubs.usgs.gov/of/2021/1030/f/ofr20211030f.pdf).

> Please note:
    `rtm_inv` does *not* provide any capabilities to query and load satellite data. We recommend to use [EOdal](https:\\github.com\EOA-Team\eodal) for this purpose.

Further sensors can be added as, both, PROSAIL and SPART output simulated spectra at a resolution of 1nm in the solar domain (400 to 2500 nm).

## Minimum Usage Example

Coming soon ...

## Work built on rtm_inv

`rtm_inv` has been used in the following scientific studies:

| Study | Purpose |  Code |
| ----- | ------- | ------ |
| [Graf et al. (2022, IEEE-JSTARS)](https://doi.org/10.1109/JSTARS.2023.3297713) | Radiometric uncertainty propagation. | [Python](https://github.com/EOA-team/s2toarup) |
| [Graf et al. (2023, RSE)](https://doi.org/10.1016/j.rse.2023.113860) | Crop trait retrieval. | [Python](https://github.com/EOA-team/sentinel2_crop_traits) |
| [Graf et al. (under review)]() | Crop trait time series reconstruction. | [Python](https://github.com/EOA-team/sentinel2_crop_trait_timeseries) |
