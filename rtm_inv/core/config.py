'''
A base class for configuring the RTM inversion and trait retrieval
'''

import numpy as np

from pathlib import Path
from typing import List, Optional

class RTMConfig:
    """
    Radiative transfer model inversion configuration set-up
    """
    def __init__(
            self,
            traits: List[str],
            rtm_params: Path,
            rtm: Optional[str] = 'prosail'
        ):
        """
        Class Constructor

        :param traits:
            list of traits to retrieve from the inversion process
            (depending on radiative transfer model chosen)
        :param rtm_params:
            CSV-file specifying RTM specific parameter ranges and their
            distribution
        :param rtm:
            name of the radiative transfer model to use. 'prosail' by
            default
        """
        self.traits = traits
        self.rtm_params = rtm_params
        self.rtm = rtm

        @property
        def rtm_params() -> Path:
            return self.rtm_params

        @property
        def traits() -> List[str]:
            return self.traits

        @property
        def rtm() -> str:
            return self.rtm

class LookupTableBasedInversion(RTMConfig):
    """
    Lookup-table based radiative transfer model inversion set-up
    """
    def __init__(
            self,
            n_solutions: int,
            cost_function: str,
            lut_size: int,
            method: Optional[str] = 'LHS',
            **kwargs
        ):
        """
        Class constructor

        :param n_solutions:
            number of solutions to use for lookup-table based inversion. Can
            be provided as relative share of spectra in the lookup-table (then
            provide a number between 0 and 1) or in absolute numbers.
        :param cost_function:
            name of the cost-function to evaluate similarity between
            observed and simulated spectra
        :param lut_size:
            size of the lookup-table (number of spectra to simulate)
        :param method:
            method to use for sampling the lookup-table
        :param kwargs:
            keyword arguments to pass to super-class
        """
        # call constructor of super-class
        super().__init__(**kwargs)
        self.cost_function = cost_function
        self.method = method
        self.lut_size = lut_size

        # adopt number of solutions if given in relative numbers [0,1[
        if 0 < n_solutions < 1:
            self.n_solutions = int(np.round(n_solutions * lut_size, 0))
        elif n_solutions < 0:
            raise ValueError('The number of solutions must not be negative')
        elif n_solutions > self.lut_size:
            raise ValueError(
                'The number of solutions must not be greater than the lookup-table size'
            )
        else:
            self.n_solutions = n_solutions

        @property
        def lut_size() -> int:
            return self.lut_size

        @property
        def n_solutions() -> int:
            return self.n_solutions

        @property
        def cost_function() -> str:
            return self.cost_function

        @property
        def method() -> str:
            return self.method
