"""
The purpose of this stage is to load in events generated from Monte Carlo
simulations.

This service in particular reads in from files having a similar structure to
the low energy event samples. More information about these event samples
can be found on
https://wiki.icecube.wisc.edu/index.php/IC86_Tau_Appearance_Analysis
https://wiki.icecube.wisc.edu/index.php/IC86_oscillations_event_selection
"""
from operator import add

import numpy as np
import pint
from uncertainties import unumpy as unp

from pisa import ureg, Q_
from pisa.core.stage import Stage
from pisa.core.map import Map, MapSet
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.utils.flavInt import NuFlavIntGroup, FlavIntDataGroup
from pisa.utils.fileio import from_file
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging
from pisa.utils.profiler import profile


class sample(Stage):
    """mc service to load in events from an event sample.

    Parameters
    ----------
    params: ParamSet of sequence with which to instantiate a ParamSet
        Parameters which set everything besides the binning

        Parameters required by this service are
            * mc_sample_config : filepath
                Filepath to event sample configuration

            * livetime : ureg.Quantity
                Desired lifetime.

    output_binning : MultiDimBinning or convertible thereto
        The binning desired for the output maps.

    transform_groups : string
        Specifies which particles/interaction types to use for computing the
        transforms.

    error_method : None, bool, or string
        If None, False, or empty string, the stage does not compute errors for
        the transforms and does not apply any (additional) error to produce its
        outputs. (If the inputs already have errors, these are propagated.)

    debug_mode : None, bool, or string
        If None, False, or empty string, the stage runs normally.
        Otherwise, the stage runs in debug mode. This disables caching (forcing
        recomputation of any nominal transforms, transforms, and outputs).

    transforms_cache_depth
    outputs_cache_depth : int >= 0

    Output Names
    ----------
    The `outputs` container generated by this service will be objects with the
    following `name` attribute:
        * 'nue_cc+nuebar_cc'
        * 'numu_cc+numubar_cc'
        * 'nutau_cc+nutaubar_cc'
        * 'nuall_nc+nuallbar_nc'
        * 'muongun'
        * 'noise'

    """
    def __init__(self, params, output_binning, output_names,
                 error_method=None, debug_mode=None, disk_cache=None,
                 memcache_deepcopy=True, transforms_cache_depth=20,
                 outputs_cache_depth=20):
        self.sample_hash = None
        """Hash of event sample"""

        expected_params = (
            'mc_sample_config', 'livetime', 'weight'
        )

        self.neutrino = False
        self.muongun = False
        self.noise = False

        output_names = output_names.replace(' ','').split(',')
        self._clean_outnames = []
        self._output_nu_groups = []
        for name in output_names:
            if 'muongun' in name:
                self.muongun = True
                self._clean_outnames.append(name)
            elif 'noise' in name:
                self.noise = True
                self._clean_outnames.append(name)
            else:
                self.neutrino = True
                self._output_nu_groups.append(NuFlavIntGroup(name))

        if self.neutrino:
            self._clean_outnames += [str(f) for f in self._output_nu_groups]

        super(self.__class__, self).__init__(
            use_transforms=False,
            params=params,
            expected_params=expected_params,
            output_names=self._clean_outnames,
            error_method=error_method,
            debug_mode=debug_mode,
            disk_cache=disk_cache,
            memcache_deepcopy=memcache_deepcopy,
            outputs_cache_depth=outputs_cache_depth,
            transforms_cache_depth=transforms_cache_depth,
            output_binning=output_binning
        )

        self.config = from_file(self.params['mc_sample_config'].value)
        self.include_attrs_for_hashes('sample_hash')

    @profile
    def _compute_outputs(self, inputs=None):
        """Compute nominal histograms for output channels."""
        self.load_sample_events()

        livetime = self.params['livetime'].to(ureg.s).m
        logging.info('Weighting with a livetime of {0} s'.format(livetime))
        outputs = []
        if self.neutrino:
            trans_nu_fidg = self._neutrino_events.transform_groups(
                self._output_nu_groups
            )
            for fig in trans_nu_fidg.iterkeys():
                if self.params['weight'].value:
                    weights = trans_nu_fidg[fig]['pisa_weight'] * \
                            livetime
                else: weights = None
                outputs.append(self._histogram(
                    events  = trans_nu_fidg[fig],
                    binning = self.output_binning,
                    weights = weights,
                    errors  = True,
                    name    = str(NuFlavIntGroup(fig)),
                ))

        if self.muongun:
            if self.params['weight'].value:
                weights = self._muongun_events['pisa_weight'] * livetime
            else: weights = None
            outputs.append(self._histogram(
                events  = self._muongun_events,
                binning = self.output_binning,
                weights = weights,
                errors  = True,
                name    = 'muongun',
                tex     = r'\rm{muongun}'
            ))

        name = self.config.get('general', 'name')
        return MapSet(maps=outputs, name=name)

    def load_sample_events(self):
        """Load the event sample given the configuration file and output
        groups. Hash this object using both the configuration file and
        the output types."""
        hash_property = [self.config, self.neutrino, self.muongun]
        this_hash = hash_obj(hash_property)
        if this_hash == self.sample_hash:
            return

        logging.info(
            'Extracting events using configuration file {0} and output names '
            '{1}'.format(hash_property[0], hash_property[1])
        )
        def parse(string):
            return string.replace(' ', '').split(',')
        event_types = parse(self.config.get('general', 'event_type'))

        # TODO(shivesh): when created, use a more generic Events object
        # (that natively supports muons, noise etc.) to store the event
        # sample
        if self.neutrino:
            if 'neutrino' not in event_types:
                raise AssertionError('`neutrino` field not found in '
                                     'configuration file.')
            self._neutrino_events = self.load_neutrino_events(
                config=self.config,
            )
        if self.muongun:
            if 'muongun' not in event_types:
                raise AssertionError('`muongun` field not found in '
                                     'configuration file.')
            self._muongun_events = self.load_moungun_events(
                config=self.config,
            )
        self.sample_hash = this_hash

    @staticmethod
    def load_neutrino_events(config):
        def parse(string):
            return string.replace(' ', '').split(',')
        flavours = parse(config.get('neutrino', 'flavours'))
        weights = parse(config.get('neutrino', 'weights'))
        sys_list = parse(config.get('neutrino', 'sys_list'))
        base_suffix = config.get('neutrino', 'basesuffix')
        if base_suffix == 'None': base_suffix = ''

        nu_fidg = []
        for idx, flav in enumerate(flavours):
            f = int(flav)
            cc_grps = NuFlavIntGroup(NuFlavIntGroup(f,-f).ccFlavInts())
            nc_grps = NuFlavIntGroup(NuFlavIntGroup(f,-f).ncFlavInts())
            flav_fidg = FlavIntDataGroup(
                flavint_groups=[cc_grps, nc_grps]
            )
            prefixes = []
            for sys in sys_list:
                ev_sys = 'neutrino:' + sys
                nominal = config.get(ev_sys, 'nominal')
                ev_sys_nom = ev_sys + ':' + nominal
                prefixes.append(config.get(ev_sys_nom, 'file_prefix'))
            if len(set(prefixes)) > 1:
                raise AssertionError(
                    'Choice of nominal file is ambigous. Nominal '
                    'choice of systematic parameters must coincide '
                    'with one and only one file. Options found are: '
                    '{0}'.format(prefixes)
                )
            file_prefix = flav + prefixes[0]
            events_file = config.get('general', 'datadir') + \
                    base_suffix + file_prefix

            events = from_file(events_file)
            cc_mask = events['ptype'] > 0
            nc_mask = events['ptype'] < 0

            if weights[idx] == 'None':
                events['pisa_weight'] = \
                        np.ones(events['ptype'].shape)
            elif weights[idx] == '0':
                events['pisa_weight'] = \
                        np.zeros(events['ptype'].shape)
            else:
                events['pisa_weight'] = events[weights[idx]]

            flav_fidg[cc_grps] = {var: events[var][cc_mask]
                                  for var in events.iterkeys()}
            flav_fidg[nc_grps] = {var: events[var][nc_mask]
                                  for var in events.iterkeys()}
            nu_fidg.append(flav_fidg)
        nu_fidg = reduce(add, nu_fidg)

        return nu_fidg

    @staticmethod
    def load_moungun_events(config):
        def parse(string):
            return string.replace(' ', '').split(',')
        sys_list = parse(config.get('muongun', 'sys_list'))
        weight = config.get('muongun', 'weight')
        base_suffix = config.get('muongun', 'basesuffix')
        if base_suffix == 'None': base_suffix = ''

        paths = []
        for sys in sys_list:
            ev_sys = 'muongun:' + sys
            nominal = config.get(ev_sys, 'nominal')
            ev_sys_nom = ev_sys + ':' + nominal
            paths.append(config.get(ev_sys_nom, 'file_path'))
        if len(set(paths)) > 1:
            raise AssertionError(
                'Choice of nominal file is ambigous. Nominal '
                'choice of systematic parameters must coincide '
                'with one and only one file. Options found are: '
                '{0}'.format(paths)
            )
        file_path = paths[0]

        muongun = from_file(file_path)

        if weight == 'None':
            muongun['pisa_weight'] = \
                    np.ones(muongun['weights'].shape)
        elif weight == '0':
            muongun['pisa_weight'] = \
                    np.zeros(muongun['weights'].shape)
        else:
            muongun['pisa_weight'] = muongun[weight]
        return muongun

    @staticmethod
    def _histogram(events, binning, weights=None, errors=False, **kwargs):
        """Histogram the events given the input binning."""
        if isinstance(binning, OneDimBinning):
            binning = MultiDimBinning([binning])
        elif not isinstance(binning, MultiDimBinning):
            raise TypeError('Unhandled type %s for `binning`.' %type(binning))
        if not isinstance(events, dict):
            raise TypeError('Unhandled type %s for `events`.' %type(events))

        bin_names = binning.names
        bin_edges = [edges.m for edges in binning.bin_edges]
        for name in bin_names:
            if not events.has_key(name):
                if 'coszen' in name and events.has_key('zenith'):
                    events[name] = np.cos(events['zenith'])
                else:
                    raise AssertionError('Input events object does not have '
                                         'key {0}'.format(name))

        sample = [events[colname] for colname in bin_names]
        hist, edges = np.histogramdd(
            sample=sample, weights=weights, bins=bin_edges
        )
        if errors:
            hist2, edges = np.histogramdd(
                sample=sample, weights=np.square(weights), bins=bin_edges
            )
            hist = unp.uarray(hist, np.sqrt(hist2))

        return Map(hist=hist, binning=binning, **kwargs)

    def validate_params(self, params):
        assert isinstance(params['mc_sample_config'].value, basestring)
        assert isinstance(params['weight'].value, bool)
        assert isinstance(params['livetime'].value, pint.quantity._Quantity)
