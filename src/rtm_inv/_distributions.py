'''
Distributions for sampling RTM parameters
'''

import numpy as np

from typing import List, Union


class Distributions(object):
    """
    Class with statistical distributions for drawning RTM
    samples from a set of input parameters

    For each RTM parameter, min, max and type of distribution
    must be passed
    """

    distributions: List[str] = ['Gaussian', 'Uniform']

    def __init__(
            self,
            min_value: Union[int, float],
            max_value: Union[int, float]
        ):
        """
        Creates a new ``Distributions`` class to use for sampling
        RTM parameter values

        :param min_value:
            minimum parameter value
        :param max_value:
            maximum parameter value
        """
        self.min_value = min_value
        self.max_value = max_value

    def sample(self,
               distribution: str,
               n_samples: int
            ):
        """
        Returns a ``numpy.ndarray`` with RTM parameter samples drawn from
        a specific distribution

        :param distribution:
            name of the distribution from which to sample. See
            `~rtm_inv.distributions.Distributions.distributions` for a list
            of distributions currently implemented
        :param n_samples:
            number of samples to draw.
        """
        if distribution == 'Uniform':
            return np.random.uniform(
                low=self.min_value,
                high=self.max_value,
                size=n_samples
            )

