"""
A Stage to load data from a CSV datarelease format file into a PISA pi ContainerSet
"""

from __future__ import absolute_import, print_function, division

import numpy as np
import pandas as pd

from pisa import FTYPE
from pisa.core.pi_stage import PiStage
from pisa.utils import vectorizer
from pisa.utils.profiler import profile
from pisa.core.container import Container
from pisa.utils.resources import find_resource


class csv_loader(PiStage):
    """
    CSV file loader PISA Pi class

    Parameters
    ----------

    events_file : csv file path

    """
    def __init__(self,
                 events_file,
                 data=None,
                 params=None,
                 input_names=None,
                 output_names=None,
                 debug_mode=None,
                 input_specs=None,
                 calc_specs=None,
                 output_specs=None,
                ):

        # instantiation args that should not change
        self.events_file = events_file

        expected_params = ()
        # created as ones if not already present
        input_apply_keys = (
            'initial_weights',
        )
        # copy of initial weights, to be modified by later stages
        output_apply_keys = (
            'weights',
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
            output_apply_keys=output_apply_keys,
        )

        # doesn't calculate anything
        if self.calc_mode is not None:
            raise ValueError(
                'There is nothing to calculate for this event loading service.'
                ' Hence, `calc_mode` must not be set.'
            )
        # check output names
        if len(self.output_names) != len(set(self.output_names)):
            raise ValueError(
                'Found duplicates in `output_names`, but each name must be'
                ' unique.'
            )

        assert self.input_specs == 'events'

    def setup_function(self):

        raw_data = pd.read_csv( find_resource(self.events_file) )

        # create containers from the events
        for name in self.output_names:

            # make container
            container = Container(name)
            container.data_specs = self.input_specs

            # get particle ID
            nubar = -1 if 'bar' in name else 1
            if 'e' in name:
                flav = 0
            if 'mu' in name:
                flav = 1
            if 'tau' in name:
                flav = 2
            pdg = nubar * (12 + 2 * flav)

            # build a mask for this flavor-interction combination
            # handling backwards compatiblity (DeepCore vs Upgrade releases)
            mask = raw_data['pdg'] == pdg
            if "current_type" in raw_data :
                mask = np.logical_and(mask, raw_data['current_type'] == (0 if name.endswith("nc") else 1) ) #TODO Handle both cases
            elif "interaction" in raw_data :
                if 'cc' in name:
                    mask = np.logical_and(mask, raw_data['type'] > 0)
                else:
                    mask = np.logical_and(mask, raw_data['type'] == 0)
            else :
                raise IOError("Could not determine NC vs CC from events in this input file")

            events = raw_data[mask]

            # Get the variables
            container['weighted_aeff'] = events['weight'].values.astype(FTYPE)
            container['weights'] = np.ones(container.array_length, dtype=FTYPE)
            container['initial_weights'] = np.ones(container.array_length, dtype=FTYPE)
            container['true_energy'] = events['true_energy'].values.astype(FTYPE)
            container['true_coszen'] = events['true_coszen'].values.astype(FTYPE) if "true_coszen" in events else np.cos(events['true_zenith'].values.astype(FTYPE))
            container['reco_coszen'] = events['reco_coszen'].values.astype(FTYPE) if "reco_coszen" in events else np.cos(events['reco_zenith'].values.astype(FTYPE))
            container['pid'] = events['pid'].values.astype(FTYPE)
            container.add_scalar_data('nubar', nubar)
            container.add_scalar_data('flav', flav)

            # azimuth is optional
            if 'true_azimuth' in events :
                container['true_azimuth'] = events['true_azimuth'].values.astype(FTYPE)
                container['reco_azimuth'] = events['reco_azimuth'].values.astype(FTYPE)

            # inelasticity is optional
            assert not ( ("inelasticity" in events) and ("y" in events) ), "Found inelasticity twice"
            if "inelasticity" in events :
                container['inelasticity'] = events['inelasticity'].values.astype(FTYPE)
            elif "y" in events :
                container['inelasticity'] = events['y'].values.astype(FTYPE)

            self.data.add_container(container)

        # check created at least one container
        if len(self.data.names) == 0:
            raise ValueError(
                'No containers created during data loading for some reason.'
            )

        # test
        if self.output_mode == 'binned':
            for container in self.data:
                container.array_to_binned('weights', self.output_specs)

    @profile
    def apply_function(self):
        for container in self.data:
            vectorizer.set(container['initial_weights'],
                           out=container['weights'])
