"""
Stage to generate some random data
"""

from __future__ import absolute_import, print_function, division

import numpy as np

from pisa import FTYPE
from pisa.core.container import Container
from pisa.core.pi_stage import PiStage
from pisa.utils import vectorizer


class toy_event_generator(PiStage):
    """
    random toy event generator PISA Pi class

    Parameters
    ----------
    data
    params
        Expected params .. ::

            n_events : int
                Number of events to be generated per output name
            random
            seed : int
                Seed to be used for random

    input_names
    output_names
    debug_mode
    input_specs
    calc_specs
    output_specs

    """
    def __init__(
        self,
        data=None,
        params=None,
        input_names=None,
        output_names=None,
        debug_mode=None,
        input_specs=None,
        calc_specs=None,
        output_specs=None,
    ):
        expected_params = ('n_events', 'random', 'seed')

        input_apply_keys = (
            'initial_weights',
            'weights',
            'weighted_aeff',
            'nu_flux_nominal',
            'nubar_flux_nominal',
        )

        # init base class
        super().__init__(
            data=data,
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            debug_mode=debug_mode,
            input_specs=input_specs,
            calc_specs=calc_specs,
            output_specs=output_specs,
            input_apply_keys=input_apply_keys,
        )

        # doesn't calculate anything
        assert self.calc_mode is None

    def setup_function(self):
        n_events = int(self.params.n_events.value.m)
        seed = int(self.params.seed.value.m)
        self.random_state = np.random.RandomState(seed)

        for name in self.output_names:

            container = Container(name)
            container.data_specs = self.input_specs
            nubar = -1 if 'bar' in name else 1
            if 'e' in name:
                flav = 0
            if 'mu' in name:
                flav = 1
            if 'tau' in name:
                flav = 2

            # Generate some events in the array representation just to have them
            # here we add those explicitly in the array representation
            true_energy = np.power(10, self.random_state.rand(n_events).astype(FTYPE) * 3)
            true_coszen = self.random_state.rand(n_events).astype(FTYPE) * 2 - 1
            container.add_array_data('true_energy', true_energy)
            container.add_array_data('true_coszen', true_coszen)

            if self.input_mode == 'events':
                size = n_events
            elif self.input_mode == 'binned':
                size = self.input_specs.size

            # make some initial weights
            if self.params.random.value:
                container['initial_weights'] = self.random_state.rand(size).astype(FTYPE)
            else:
                container['initial_weights'] = np.ones(size, dtype=FTYPE)

            # other necessary info
            container.add_scalar_data('nubar', nubar)
            container.add_scalar_data('flav', flav)
            container['weights'] = np.ones(size, dtype=FTYPE)
            container['weighted_aeff'] = np.ones(size, dtype=FTYPE)

            flux_nue = np.zeros(size, dtype=FTYPE)
            flux_numu = np.ones(size, dtype=FTYPE)
            flux = np.stack([flux_nue, flux_numu], axis=1)

            container['nu_flux_nominal'] = flux
            container['nubar_flux_nominal'] = flux

            self.data.add_container(container)

    def apply_function(self):
        # reset weights
        for container in self.data:
            vectorizer.assign(container['initial_weights'], out=container['weights'])
