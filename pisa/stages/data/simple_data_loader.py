"""
A Stage to load data from a PISA style hdf5 file into a PISA pi ContainerSet
"""
from __future__ import absolute_import, print_function, division

import numpy as np

from pisa import FTYPE
from pisa.core.pi_stage import PiStage
from pisa.utils import vectorizer
from pisa.utils.profiler import profile
from pisa.core.container import Container
from pisa.core.events_pi import EventsPi


class simple_data_loader(PiStage):
    """
    HDF5 file loader PISA Pi class

    Parameters
    ----------

    events_file : hdf5 file path
        output from make_events, including flux weights
        and Genie systematics coefficients

    mc_cuts : cut expr
        e.g. '(true_coszen <= 0.5) & (true_energy <= 70)'

    data_dict : str of a dict
        Dictionary to specify what keys from the hdf5 files to be loaded
        under what name. Entries can be strings that point to the right
        key in the hdf5 file or lists of keys, and the data will be
        stacked into a 2d array.

    Notes
    -----
    Looks for `initial_weights` fields in events file, which will serve
    as nominal weights for all events included.
    No fields named `weights` may already be present.

    """
    def __init__(self,
                 events_file,
                 mc_cuts,
                 data_dict,
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
        self.mc_cuts = mc_cuts
        self.data_dict = data_dict

        # instead of adding params here, consider making them instantiation
        # args so nothing external will inadvertently try to change
        # their values
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
        super(simple_data_loader, self).__init__(
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
        if len(self.output_names) == 0:
            raise ValueError(
                'Must specify at least one output element in `output_names`.'
            )
        if len(self.output_names) != len(set(self.output_names)):
            raise ValueError(
                'Found duplicates in `output_names`, but each name must be'
                ' unique.'
            )

        self.load_events()
        self.apply_cuts_to_events()

    def load_events(self):
        '''Loads events from events file'''
        self.evts = EventsPi(name='Events')
        self.data_dict = eval(self.data_dict)
        self.evts.load_events_file(
            events_file=self.events_file,
            variable_mapping=self.data_dict
        )

    def apply_cuts_to_events(self):
        '''Just apply any cuts that the user defined'''
        if self.mc_cuts:
            self.evts = self.evts.apply_cut(self.mc_cuts)

    def record_event_properties(self):
        '''Adds fields present in events file and selected in `self.data_dict`
        into containers for the specified output names. Also ensures the
        presence of a set of nominal weights.
        '''
        # create containers from the events
        for name in self.output_names:
            # make container
            container = Container(name)
            container.data_specs = 'events'
            event_groups = self.evts.keys()
            if name not in event_groups:
                raise ValueError(
                    'Output name "%s" not found in events. Only found %s.'
                    % (name, event_groups)
                )
            # add the events data to the container
            for key, val in self.evts[name].items():
                container.add_array_data(key, val)
            # create weights arrays:
            # * `initial_weights` as starting point (never modified)
            # * `weights` to be initialised from `initial_weights`
            #   and modified by the stages
            # * user can also provide `initial_weights` in input file
            if 'weights' in container.array_data:
                # raise manually to give user some helpful feedback
                raise KeyError(
                    'Found an existing `weights` array in "%s"'
                    ' which would be overwritten. Consider renaming it'
                    ' to `initial_weights`.' % name
                )
            container.add_array_data(
                'weights',
                np.ones(container.size, dtype=FTYPE)
            )
            if 'initial_weights' not in container.array_data:
                container.add_array_data(
                    'initial_weights',
                    np.ones(container.size, dtype=FTYPE)
                )
            # add neutrino flavor information for neutrino events
            if name.startswith('nu'):
                # this determination of flavour is the worst possible coding
                # TODO
                nubar = -1 if 'bar' in name else 1
                if 'tau' in name:
                    flav = 2
                elif 'mu' in name:
                    flav = 1
                elif 'e' in name:
                    flav = 0
                else:
                    raise ValueError('Cannot determine flavor of "%s".' % name)
                container.add_scalar_data('nubar', nubar)
                container.add_scalar_data('flav', flav)

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

    def setup_function(self):
        '''Store event properties from events file at
        service initialisation. Cf. `PiStage` docs.
        '''
        self.record_event_properties()

    @profile
    def apply_function(self):
        '''Cf. `PiStage` docs.'''
        # TODO: do we need following line? Isn't this handled universally
        # by the base class (in PiStage's apply)?
        self.data.data_specs = self.output_specs
        # reset weights to initial weights prior to downstream stages running
        for container in self.data:
            vectorizer.set(container['initial_weights'],
                           out=container['weights'])