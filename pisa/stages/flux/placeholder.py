# pylint: disable=not-callable, wrong-import-position


from __future__ import absolute_import, print_function, division

import math
import os
import sys

import numpy as np
from numba import guvectorize, cuda

from pisa import FTYPE, TARGET
from pisa.core.pi_stage import PiStage
from pisa.utils.profiler import profile
from pisa.utils.numba_tools import WHERE, myjit, ftype
from pisa.utils.resources import find_resource

class placeholder(PiStage):  # pylint: disable=invalid-name
    """
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
        expected_params = (
            "nue_flux_spectral_index",
            "numu_flux_spectral_index",
            "nutau_flux_spectral_index",
            "nue_flux_norm",
            "numu_flux_norm",
            "nutau_flux_norm",
        )
        input_names = ()
        output_names = ()

        # what are the keys used from the inputs during apply
        input_calc_keys = ()
        # what are keys added or altered in the calculation used during apply
        output_calc_keys = ("flux",)
        # what keys are added or altered for the outputs during apply
        output_apply_keys = ("weights",)

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
            input_calc_keys=input_calc_keys,
            output_calc_keys=output_calc_keys,
            output_apply_keys=output_apply_keys,
        )

        assert self.input_mode == "events"
        assert self.calc_mode == "events"
        assert self.output_mode == "events"


    def setup_function(self):
        self.data.data_specs = self.input_specs
        for container in self.data:
            container["flux"] = np.empty(container.size, dtype=FTYPE)


    @profile
    def compute_function(self):
        self.data.data_specs = self.calc_specs

        nue_flux_spectral_index = self.params.nue_flux_spectral_index.value.m_as("dimensionless")
        numu_flux_spectral_index = self.params.numu_flux_spectral_index.value.m_as("dimensionless")
        nutau_flux_spectral_index = self.params.nutau_flux_spectral_index.value.m_as("dimensionless")
        nue_flux_norm = self.params.nue_flux_norm.value.m_as("dimensionless")
        numu_flux_norm = self.params.numu_flux_norm.value.m_as("dimensionless")
        nutau_flux_norm = self.params.nutau_flux_norm.value.m_as("dimensionless")

        for container in self.data:

            if container.name.startswith("nue") :
                norm = nue_flux_norm
                spectral_index = nue_flux_spectral_index
            elif container.name.startswith("numu") :
                norm = numu_flux_norm
                spectral_index = numu_flux_spectral_index
            elif container.name.startswith("nutau") :
                norm = nutau_flux_norm
                spectral_index = nutau_flux_spectral_index
            else :
                raise Exception("Unrecognised data category : %s" % container.name)

            flux = norm * np.power( container["true_energy"].get("host"), spectral_index ) 
            np.copyto( src=flux, dst=container["flux"].get("host") )
            container["flux"].mark_changed("host")


    @profile
    def apply_function(self):
        self.data.data_specs = self.output_specs

        # update the outputted weights
        for container in self.data:
            np.copyto( src=container["flux"].get("host"), dst=container["weights"].get("host") )
            container["weights"].mark_changed("host")

