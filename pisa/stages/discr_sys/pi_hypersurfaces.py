"""
PISA pi stage to apply hypersurface fits from discrete systematics parameterizations
"""

from __future__ import absolute_import, print_function, division

import ast

from numba import guvectorize
import numpy as np

from pisa import FTYPE, TARGET, ureg
from pisa.core.binning import MultiDimBinning
from pisa.core.pi_stage import PiStage
from pisa.utils.fileio import from_file
from pisa.utils.log import logging
from pisa.utils.numba_tools import WHERE
from pisa.utils import vectorizer
from pisa.utils.hypersurface import load_hypersurfaces
from numba import guvectorize

__all__ = ["pi_hypersurfaces",]

__author__ = "P. Eller, T. Ehrhardt, T. Stuttard, J.L. Lanfranchi"

__license__ = """Copyright (c) 2014-2018, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License."""


# TODO: consider taking into account fit parameter covariances
class pi_hypersurfaces(PiStage):  # pyint: disable=invalid-name
    """
    Service to apply hypersurface parameterisation produced by
    `scripts.fit_discrete_sys_nd`

    Parameters
    ----------
    fit_results_file : str
        Path to hypersurface fit results file, i.e. the JSON file produced by the
        `pisa.scripts.fit_discrete_sys_nd.py` script

    params : ParamSet
        Note that the params required to be in `params` are found from
        those listed in the `fit_results_file`

    Notes
    -----
    TODO

    """

    def __init__(
        self,
        fit_results_file,
        data=None,
        params=None,
        input_names=None,
        output_names=None,
        debug_mode=None,
        error_method=None,
        input_specs=None,
        calc_specs=None,
        output_specs=None,
        links=None,
    ):
        # -- Read fit_results_file and extract necessary info -- #

        # Store args
        self.fit_results_file = fit_results_file

        # Load hypersurfaces
        self.hypersurfaces = load_hypersurfaces(self.fit_results_file)

        # Get the expected param names
         #TODO change name from `fit` params
        self.hypersurface_param_names = list(self.hypersurfaces.values())[0].param_names

        # -- Expected input / output names -- #
        input_names = ()
        output_names = ()

        # -- Which keys are added or altered for the outputs during `apply` -- #

        input_calc_keys = ()
        output_calc_keys = ("hypersurface_scalefactors",)

        if error_method == "sumw2":
            output_apply_keys = ("weights", "errors")
            input_apply_keys = output_apply_keys
        else:
            output_apply_keys = ("weights",)
            input_apply_keys = output_apply_keys

        # -- Initialize base class -- #

        super(pi_hypersurfaces, self).__init__(
            data=data,
            params=params,
            expected_params=self.hypersurface_param_names,
            input_names=input_names,
            output_names=output_names,
            debug_mode=debug_mode,
            error_method=error_method,
            input_specs=input_specs,
            calc_specs=calc_specs,
            output_specs=output_specs,
            input_calc_keys=input_calc_keys,
            output_calc_keys=output_calc_keys,
            input_apply_keys=input_apply_keys,
            output_apply_keys=output_apply_keys,
        )

        # -- Only allowed/implemented modes -- #

        assert self.input_mode is not None
        assert self.calc_mode == "binned"
        assert self.output_mode is not None

        self.links = ast.literal_eval(links) #TODO directly use compile_regex from hypersurface?


    def setup_function(self):
        """Load the fit results from the file and make some check compatibility"""

        self.data.data_specs = self.calc_specs

        if self.links is not None: #TODO use stored compare_regex
            for key, val in self.links.items():
                self.data.link_containers(key, val)

        # create containers for scale factors
        for container in self.data :
            container["hypersurface_scalefactors"] = np.empty(container.size, dtype=FTYPE)

        # Check binning compatibility
        #TODO binning hash

        # Check params match
        #TODO

        # Check map names match
        #TODO

        # Check nominal values match between the stage params and the hypersurface params
        #TODO

        self.data.unlink_containers()

    def compute_function(self):

        self.data.data_specs = self.calc_specs

        # Link containers
        if self.links is not None:
            for key, val in self.links.items():
                self.data.link_containers(key, val)

        # Format the params dict
        #TODO handle units
        # param_values = { sys_param_name: self.params[sys_param_name].m_as(units) for sys_param_name, units in zip(self.hypersurface_param_names, self.fit_param_units) } #TODO change name from fit params
        param_values = { sys_param_name: self.params[sys_param_name].m for sys_param_name in self.hypersurface_param_names }

        # Evaluate the hypersurfaces
        for container in self.data:

            # Get the hypersurface scale factors
            # Reshape to 1D array
            scalefactors = self.hypersurfaces[container.name].evaluate(param_values).reshape(container.size)

            # Where there are no scalefactors (e.g. empty bins), set scale factor to 1 
            #TODO maybe this should be handle by Hyperplane.evaluate directly??
            scalefactors[~np.isfinite(scalefactors)] = 1.
            
            # Add to container
            #TODO Directly modify the container in the first place
            np.copyto( src=scalefactors, dst=container["hypersurface_scalefactors"].get(WHERE) )
            container["hypersurface_scalefactors"].mark_changed()
            #TODO verctorise, get(WHERE), mark_changed, etc

        # Unlink the containers again
        self.data.unlink_containers()


    def apply_function(self):

        for container in self.data:

            # Update weights according to hypersurfaces
            vectorizer.multiply(
                container["hypersurface_scalefactors"], container["weights"]
            )

            if self.error_method == "sumw2":
                vectorizer.multiply(
                    container["hypersurface_scalefactors"], container["errors"]
                )

            # Correct negative event counts that can be introduced by hypersurfaces (due to intercept)
            #TODO probably can make this more efficient
            weights = container["weights"].get(WHERE)
            neg_mask = weights < 0.
            if neg_mask.sum() > 0 :
                weights[neg_mask] = 0.
                np.copyto( src=weights, dst=container["weights"].get(WHERE) )
                container["weights"].mark_changed()
