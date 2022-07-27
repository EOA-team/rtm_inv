'''
Module to create lookup-tables (LUT) of synthetic spectra
'''

import lhsmdu
import numpy as np
import pandas as pd

from pathlib import Path
from typing import List, Optional, Union

from rtm_inv.core.distributions import Distributions
from rtm_inv.core.rtm_adapter import RTM
from numba.core.types import none

sampling_methods: List[str] = ['LHS']

class LookupTable(object):
    """
    Lookup-table with RTM simulated spectra plus corresponding
    parameterization (leaf and canopy traits)

    :attrib samples:
        RTM trait samples generated using a custom sample strategy
        sampling. RTM-generated spectra are appended as additional
        columns.
    """
    def __init__(
            self,
            params: Union[Path,pd.DataFrame]
        ):
        """
        creates a new ``Lookup Table`` instance

        :param params:
            csv file with RTM parameters (traits), their min and max
            value and selected distribution
        """
        if isinstance(params, Path):
            self._params_df = pd.read_csv(self.params_csv)
        elif isinstance(params, pd.DataFrame):
            self._params_df = params.copy()
        else:
            raise TypeError('Expected Path-object or DataFrame')
        self.samples = None

    @property
    def samples(self) -> Union[pd.DataFrame, None]:
        """
        Trait samples for generating synthetic spectra
        """
        return self._samples

    @samples.setter
    def samples(self, in_df: pd.DataFrame) -> None:
        """
        Trait samples for generating synthetic spectra
        """
        if in_df is not None:
            if not isinstance(in_df, pd.DataFrame):
                raise TypeError(
                    f'Expected a pandas DataFrame instance, got {type(in_df)}'
                )
        self._samples = in_df

    def generate_samples(
            self,
            num_samples: int,
            method: str,
            seed_value: Optional[int] = 0
        ):
        """
        Sample parameter values using a custom sampling scheme.

        Currently supported sampling schemes are:

        - Latin Hypercube Sampling (LHS)
        - Fully Random Sampling (FRS)
        - ...

        All parameters (traits) are sampled, whose distribution is not set
        as "constant"

        :param num_samples:
            number of samples to draw (equals the size of the resulting
            lookup-table)
        :param method:
            sampling method to apply
        :param seed_value:
            seed value to set to the pseudo-random-number generator. Default
            is zero.
        """
        # set seed to the random number generator
        np.random.seed(seed_value)

        # determine traits to sample (indicated by a distribution different from
        # "Constant"
        traits = self._params_df[
            self._params_df['Distribution'].isin(Distributions.distributions)
        ]
        trait_names = traits['Parameter'].to_list()
        traits = traits.transpose()
        traits.columns = trait_names
        n_traits = len(trait_names)

        # and those traits/ parameters remaining constant
        constant_traits = self._params_df[
            ~self._params_df['Parameter'].isin(trait_names)
        ]
        constant_trait_names = constant_traits['Parameter'].to_list()
        constant_traits = constant_traits.transpose()
        constant_traits.columns = constant_trait_names

        # select method and conduct sampling
        if method.upper() == 'LHS':
            # create LHS matrix
            lhc = lhsmdu.createRandomStandardUniformMatrix(n_traits, num_samples)
            traits_lhc = pd.DataFrame(lhc).transpose()
            traits_lhc.columns = trait_names
            # replace original values in LHS matrix (scaled between 0 and 1) with
            # trait values scaled between their specific min and max
            for trait_name in trait_names:
                traits_lhc[trait_name] = traits_lhc[trait_name] * \
                    traits[trait_name]['Max'] + traits[trait_name]['Min']
            sample_traits = traits_lhc
        elif method.upper() == 'FRS':
            # fully random sampling within the trait ranges specified
            # drawing a random sample for each trait
            frs_matrix = np.empty((num_samples, n_traits), dtype=np.float32)
            traits_frs = pd.DataFrame(frs_matrix)
            traits_frs.columns = trait_names
            for trait_name in trait_names:
                mode = None
                if 'Mode' in traits[trait_name].index:
                    mode = traits[trait_name]['Mode']
                std = None
                if 'Std' in traits[trait_name].index:
                    std = traits[trait_name]['Std']
                dist = Distributions(
                    min_value=traits[trait_name]['Min'],
                    max_value=traits[trait_name]['Max'],
                    mean_value=mode,
                    std_value=std
                )
                traits_frs[trait_name] = dist.sample(
                    distribution=traits[trait_name]['Distribution'],
                    n_samples=num_samples
                )
            sample_traits = traits_frs
        else:
            raise NotImplementedError(f'{method} not found')

        # combine trait samples and constant values into a single DataFrame
        # so that in can be passed to the RTM
        for constant_trait in constant_trait_names:
            # for constant traits the value in the min column is used
            # (actually min and max should be set to the same value)
            sample_traits[constant_trait] = constant_traits[constant_trait]['Min']

        # set samples to instance variable
        self.samples = sample_traits

def _setup(
        lut_params: pd.DataFrame,
        rtm_name: str,
        solar_zenith_angle: Optional[float] = None,
        viewing_zenith_angle: Optional[float] = None,
        solar_azimuth_angle: Optional[float] = None,
        viewing_azimuth_angle: Optional[float] = None
    ) -> pd.DataFrame:
    """
    Setup LUT for RTM (modification of angles and names if required)
    """
    if rtm_name == 'prosail':
        sol_angle = 'tts'   # solar zenith
        obs_angle = 'tto'   # observer zenith
        rel_angle = 'psi'   # relative azimuth
    elif rtm_name == 'spart':
        sol_angle = 'sol_angle'
        obs_angle = 'obs_angle'
        rel_angle = 'rel_angle'

    # overwrite angles in LUT DataFrame if provided as fixed values
    if solar_zenith_angle is not None:
        lut_params.loc[lut_params['Parameter'] == sol_angle,'Min'] = solar_zenith_angle
        lut_params.loc[lut_params['Parameter'] == sol_angle,'Max'] = solar_zenith_angle
    if viewing_zenith_angle is not None:
        lut_params.loc[lut_params['Parameter'] == obs_angle, 'Min'] = viewing_zenith_angle
        lut_params.loc[lut_params['Parameter'] == obs_angle, 'Max'] = viewing_zenith_angle
    # calculate relative azimuth (psi) if viewing angles are passed
    if viewing_azimuth_angle is not None and solar_azimuth_angle is not None:
        psi = abs(solar_azimuth_angle - viewing_azimuth_angle)
        lut_params.loc[lut_params['Parameter'] == rel_angle, 'Min'] = psi
        lut_params.loc[lut_params['Parameter'] == rel_angle, 'Max'] = psi

    # 'mode' and 'std' are optional columns
    further_columns = ['Mode', 'Std']
    for further_column in further_columns:
        if further_column in lut_params.columns:
            lut_params.loc[lut_params['Parameter'] == sol_angle, further_column] = solar_zenith_angle
            lut_params.loc[lut_params['Parameter'] == obs_angle, further_column] = viewing_zenith_angle
            lut_params.loc[lut_params['Parameter'] == rel_angle, further_column] = psi

    return lut_params

def generate_lut(
        sensor: str,
        lut_params: Union[Path, pd.DataFrame],
        lut_size: Optional[int] = 50000,
        rtm_name: Optional[str] = 'prosail',
        sampling_method: Optional[str] = 'LHS',
        solar_zenith_angle: Optional[float] = None,
        viewing_zenith_angle: Optional[float] = None,
        solar_azimuth_angle: Optional[float] = None,
        viewing_azimuth_angle: Optional[float] = None,
        **kwargs
    ) -> pd.DataFrame:
    """
    Generates a Lookup-Table (LUT) based on radiative transfer model input parameters.

    IMPORTANT:
        Depending on the RTM and the size of the LUT the generation of a LUT
        might take a while!

    :param sensor:
        name of the sensor for which the simulated spectra should be resampled.
        See `rtm_inv.core.sensors.Sensors` for a list of sensors currently implemented.
    :param lut_params:
        lookup-table parameters with mininum and maximum range (always required),
        type of distribution (important to indicate which parameters are constant),
        mode and std (for Gaussian distributions).
    :param lut_size:
        number of items (spectra) to simulate in the LUT
    :param rtm_name:
        name of the RTM to call.
    :param sampling_method:
        sampling method for generating the input parameter space of the LUT. 'LHS'
        (latin hypercube sampling) by default.
    :param solar_zenith_angle:
        solar zenith angle as fixed scene-wide value (optional) in degrees.
    :param viewing_zenith_angle:
        viewing (observer) zenith angle as fixed scene-wide value (optional) in degrees.
    :param solar_azimuth_angle:
        solar azimuth angle as fixed scene-wide value (optional) in deg C.
    :param viewing_azimuth_angle:
        viewing (observer) azimuth angle as fixed scene-wide value (optional) in deg C.
    :param kwargs:
        optional keyword-arguments to pass to `LookupTable.generate_samples`
    :returns:
        input parameters and simulated spectra as `DataFrame`.
    """
    # read parameters from CSV if not provided as a DataFrame
    if isinstance(lut_params, Path):
        lut_params = pd.read_csv(lut_params)

    # prepare LUTs for RTMs
    lut_params = _setup(lut_params, rtm_name, solar_zenith_angle, viewing_zenith_angle,
                        solar_azimuth_angle, viewing_azimuth_angle)
    # get input parameter samples first
    lut = LookupTable(params=lut_params)
    lut.generate_samples(num_samples=lut_size, method=sampling_method, **kwargs)

    # and run the RTM in forward mode in the second step
    # outputs get resampled to the spectral resolution of the sensor
    rtm = RTM(lut=lut, rtm=rtm_name)
    lut_simulations = rtm.simulate_spectra(sensor=sensor)
    return lut_simulations
