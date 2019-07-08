
"""
Stage to implement the intrimsic neutrino flux as calculated with MCEq, 
and the systematic flux variations based on the Barr scheme. 

It requires spline tables created by the `$PISA/scripts/create_barr_sys_tables_mceq.py`
"""
from __future__ import absolute_import, print_function, division

import math, collections
import numpy as np
from numba import guvectorize, cuda
import cPickle as pickle
from bz2 import BZ2File
from scipy.interpolate import RectBivariateSpline

from pisa import FTYPE, TARGET
from pisa.core.pi_stage import PiStage
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.utils.numba_tools import WHERE, myjit, ftype
from pisa.utils.resources import find_resource


class pi_mceq_barr(PiStage):
    """
    stage generate nominal flux from MCEq and apply Barr style flux uncertainties.

    Paramaters
    ----------
    barr_* : quantity (dimensionless)

    Notes
    -----
    The table consists of 2 solutions of the cascade equation per Barr variable (12) 
    - one solution for meson and one solution for the antimeson. 
    Each solution consists of 8 splines: idx=0,2,4,6=numu, numubar, nue, nuebar. 
    idx=1,3,5,7=gradients of numu, numubar, nue, nuebar. 

    """

    def __init__(self,
        table_file,
        data=None,
        params=None,
        input_names=None,
        output_names=None,
        debug_mode=None,
        input_specs=None,
        calc_specs=None,
        output_specs=None,
    ):

        #
        # Define parameterisation
        # 

        # Define the Barr parameters
        self.barr_param_names = [ #TODO common code with `create_barr_sys_tables_mceq.py` ?
            # pions
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            # kaons
            "w",
            # "x", #TODO x seems to be missing from the splines?????? re-generate
            "y",
            "z",
        ]

        # Define signs for Barr params
        # +  -> meson
        # -  -> antimeson
        self.barr_param_signs = ["+","-"]

        # Atmopshere model params
        #TODO

        # Get the overall list of params for which we have gradients stored
        # Define a mapping to index values, will he useful later
        self.gradient_param_names = [ n+s for n in self.barr_param_names for s in self.barr_param_signs ]
        self.gradient_param_indices = collections.OrderedDict([ (n,i) for i,n in enumerate(self.gradient_param_names) ])


        #
        # Call stage base class constructor
        #

        # Define stage parameters
        expected_params = (
            'barr_a',
            'barr_b',
            'barr_c',
            'barr_d',
            'barr_e',
            'barr_f',
            'barr_g',
            'barr_h',
            'barr_i',
            'barr_w',
            'barr_x',
            'barr_y',
            'barr_z',
            'delta_index',
            'energy_pivot',
        )

        input_names = ()
        output_names = ()

        # what are the keys used from the inputs during apply
        input_calc_keys = ()

        # what are keys added or altered in the calculation used during apply
        output_calc_keys = ('nu_flux_nominal',
                            'nu_flux',
                            )
        # what keys are added or altered for the outputs during apply
        output_apply_keys = ('nu_flux_nominal',
                            'nu_flux',
                            )

        # store args
        self.table_file = table_file

        # init base class
        super(pi_mceq_barr, self).__init__(
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

        assert self.input_mode is not None
        assert self.calc_mode is not None
        assert self.output_mode is not None



    def setup_function(self):

        self.data.data_specs = self.calc_specs


        #
        # Init arrays
        #

        # Prepare some array shapes 
        gradient_params_shape = ( len(self.gradient_param_names), )
        
        # Loop over containers
        for container in self.data:

            # Define shapes for containers
            #TODO maybe include toggles for nutau (only needed if prompt considered) and for nu+nubar (only needed if nu->nubar oscillations included) for better speed/memory performance
            flux_container_shape = ( container.size, 3 ) # [ N events, 3 flavors in flux, nu vs nubar ]
            gradients_shape = tuple( list(flux_container_shape) + list(gradient_params_shape) )

            # Create arrays that will be populated in the stage
            # Note that the flux arrays will be chosen as nu or nubar depending on the container (e.g. not simultaneously storing nu and nubar)
            # Would rather use multi-dim arrays here but limited by fact that numba only supports 1/2D versions of numpy functions
            container['nu_flux_nominal'] = np.full( flux_container_shape, np.NaN, dtype=FTYPE ) 
            container['nu_flux'] = np.full( flux_container_shape, np.NaN, dtype=FTYPE )
            container['gradients'] = np.full( gradients_shape, np.NaN, dtype=FTYPE )
        
        # Also create an array container to hold the gradient parameter values 
        # Only want this once, e.g. not once per container
        self.gradient_params = np.empty( gradient_params_shape, dtype=FTYPE ) #TODO More efficient on GPUs if I use SmartArray?, so can support GPUs


        #
        # Load MCEq splines
        #

        '''
        Have splined both nominal fluxes and gradients in flux w.r.t. 
        Barr parameters, using MCEQ.

        Have splines for each Barr parameter, plus +/- versions of each 
        Barr parameter corresponding to mesons/antimesons.

        For a give Barr parameter, order of splines is:
            [numu, dnumu, anumu, danumu, nue, dnue, anue, danue] #TODO dict instead?

        Note that doing this all on CPUs, since the splines reside on the CPUs
        The actual `compute_function` computation can be done on GPUs though
        '''

        # Load the MCEq splines
        self.spline_tables_dict = pickle.load( BZ2File( find_resource(self.table_file) ) )

        #TODO Need to also include splines with heavy ions (so far only have protons)

        # Loop over containers
        for container in self.data :

            # Grab containers here once to save time
            true_log_energy = np.log( container["true_energy"].get("host") ) #TODO store splines directly in terms of energy, not log energy
            true_abs_coszen = np.abs( container["true_coszen"].get("host") )
            nu_flux_nominal = container["nu_flux_nominal"].get("host")
            gradients = container["gradients"].get("host")
            nubar = container["nubar"]


            #
            # Nominal flux
            #

            # Evaluate splines to get nominal flux
            # Need to correctly map nu/nubar and flavor to the output arrays

            # Note that nominal flux is stored multiple times (once per Barr parameter)
            # Choose an arbitrary one to get the nominal fluxes
            arb_gradient_param_key = self.gradient_param_names[0]

            # nue(bar)
            self._eval_spline( 
                true_log_energy=true_log_energy, 
                true_abs_coszen=true_abs_coszen, 
                spline=self.spline_tables_dict[arb_gradient_param_key][4 if nubar > 0 else 6],
                out=nu_flux_nominal[:,0],
            )

            # numu(bar)
            self._eval_spline( 
                true_log_energy=true_log_energy, 
                true_abs_coszen=true_abs_coszen, 
                spline=self.spline_tables_dict[arb_gradient_param_key][0 if nubar > 0 else 2],
                out=nu_flux_nominal[:,1],
            )


            # nutau(bar)
            # Currently setting to 0 #TODO include nutau flux (e.g. prompt) in splines
            nu_flux_nominal[:,2].fill(0.)

            # Tell the smart arrays we've changed the nominal flux values on the host
            container["nu_flux_nominal"].mark_changed("host")


            #
            # Flux gradients
            #

            # Evaluate splines to get the flux graidents w.r.t the Barr parameter values
            # Once again, need to correctly map nu/nubar and flavor to the output arrays

            # Loop over parameters
            for gradient_param_name,gradient_param_idx in self.gradient_param_indices.items() :

                # nue(bar)
                self._eval_spline( 
                    true_log_energy=true_log_energy, 
                    true_abs_coszen=true_abs_coszen, 
                    spline=self.spline_tables_dict[gradient_param_name][5 if nubar > 0 else 7],
                    out=gradients[:,0,gradient_param_idx],
                )

                # numu(bar)
                self._eval_spline( 
                    true_log_energy=true_log_energy, 
                    true_abs_coszen=true_abs_coszen, 
                    spline=self.spline_tables_dict[gradient_param_name][1 if nubar > 0 else 3],
                    out=gradients[:,1,gradient_param_idx],
                )

                # nutau(bar)
                #TODO include nutau flux in splines
                gradients[:,2,gradient_param_idx].fill(0.)

            # Tell the smart arrays we've changed the flux gradient values on the host
            container["gradients"].mark_changed("host")


    def _eval_spline(self, true_log_energy, true_abs_coszen, spline, out):
        '''
        Evaluate the spline for the full arrays of [ ln(energy), abs(coszen) ] values
        '''

        # Evalate the spine
        result = spline( true_abs_coszen, true_log_energy, grid=False )

        # Correct units
        #TODO Make MCEq spline creation script write in the desired units (assuming splining still behaves nicely...)
        result = result * 1.e4 #TODO document units

        # Copy to output array
        #TODO Can I directly write to the original array, will be faster
        np.copyto( src=result, dst=out )


    @profile
    def compute_function(self):

        self.data.data_specs = self.calc_specs

        #
        # Get params
        #

        # Spectral index (and required energy pivot)
        delta_index = self.params.delta_index.value.m_as("dimensionless")
        energy_pivot = self.params.energy_pivot.value.m_as("GeV")

        # User variants of Barr parameterisation
        barr_a = self.params.barr_a.value.m_as('dimensionless')
        barr_b = self.params.barr_b.value.m_as('dimensionless')
        barr_c = self.params.barr_c.value.m_as('dimensionless')
        barr_d = self.params.barr_d.value.m_as('dimensionless')
        barr_e = self.params.barr_e.value.m_as('dimensionless')
        barr_f = self.params.barr_f.value.m_as('dimensionless')
        barr_g = self.params.barr_g.value.m_as('dimensionless')
        barr_h = self.params.barr_h.value.m_as('dimensionless')
        barr_i = self.params.barr_i.value.m_as('dimensionless')
        barr_w = self.params.barr_w.value.m_as('dimensionless')
        barr_x = self.params.barr_x.value.m_as('dimensionless')
        barr_y = self.params.barr_y.value.m_as('dimensionless')
        barr_z = self.params.barr_z.value.m_as('dimensionless')

        # Map the user parameters into the Barr +/- params
        #TODO implement pi+/pi- ratio and K params
        gradient_params_mapping = collections.OrderedDict()
        gradient_params_mapping["a+"] = barr_a
        gradient_params_mapping["a-"] = barr_a
        gradient_params_mapping["b+"] = barr_b
        gradient_params_mapping["b-"] = barr_b
        gradient_params_mapping["c+"] = barr_c
        gradient_params_mapping["c-"] = barr_c
        gradient_params_mapping["d+"] = barr_d
        gradient_params_mapping["d-"] = barr_d
        gradient_params_mapping["e+"] = barr_e
        gradient_params_mapping["e-"] = barr_e
        gradient_params_mapping["f+"] = barr_f
        gradient_params_mapping["f-"] = barr_f
        gradient_params_mapping["g+"] = barr_g
        gradient_params_mapping["g-"] = barr_g
        gradient_params_mapping["h+"] = barr_h
        gradient_params_mapping["h-"] = barr_h
        gradient_params_mapping["i+"] = barr_i
        gradient_params_mapping["i-"] = barr_i
        gradient_params_mapping["w+"] = barr_w
        gradient_params_mapping["w-"] = barr_w
        # gradient_params_mapping["x+"] = barr_x #TODO
        # gradient_params_mapping["x-"] = barr_x
        gradient_params_mapping["y+"] = barr_y
        gradient_params_mapping["y-"] = barr_y
        gradient_params_mapping["z+"] = barr_z
        gradient_params_mapping["z-"] = barr_z

        # Populate array Barr param array
        for gradient_param_name,gradient_param_idx in self.gradient_param_indices.items() :
            assert gradient_param_name in gradient_params_mapping, "Gradient parameter '%s' missing from mapping" % gradient_param_name
            self.gradient_params[gradient_param_idx] = gradient_params_mapping[gradient_param_name]


        #
        # Loop over containers
        #

        for container in self.data:

            #
            # Apply the systematics to the flux
            #

            apply_sys_vectorized(
                container["true_energy"].get(WHERE),
                container["true_coszen"].get(WHERE),
                delta_index,
                energy_pivot,
                container['nu_flux_nominal'].get(WHERE),
                container['gradients'].get(WHERE),
                self.gradient_params,
                out=container['nu_flux'].get(WHERE),
            )

            container['nu_flux'].mark_changed(WHERE)

            # Check for negative results from spline
            #TODO


@myjit
def spectral_index_scale(true_energy, energy_pivot, delta_index):
      """
      Calculate spectral index scale.
      Adjusts the weights for events in an energy dependent way according to a 
      shift in spectral index, applied about a user-defined energy pivot.
      """
      return np.power( (true_energy / energy_pivot), delta_index)


@myjit
def apply_sys_kernel(
    true_energy,
    true_coszen,
    delta_index,
    energy_pivot,
    nu_flux_nominal,
    gradients,
    gradient_params,
    out,
):
    '''
    Calculation:
      1) Start from nominal flux
      2) Apply spectral index shift #TODO Do this BEFORE or AFTER Hadronic contribution?
      3) Add contributions from MCEq-computed gradients

    Array dimensions :
        true_energy : [A]
        true_coszen : [A]
        nubar : scalar integer
        delta_index : scalar float
        nu_flux_nominal : [A,B]
        gradients : [A,B,C]
        gradient_params : [C] 
        out : [A,B] (sys flux)
    where:
        A = num events
        B = num flavors in flux (=3, e.g. e, mu, tau)
        C = num gradients
    Not that first dimension (of length A) is vectorized out
    '''
    out[...] = ( nu_flux_nominal * spectral_index_scale(true_energy, energy_pivot, delta_index) ) + np.dot(gradients,gradient_params)


# vectorized function to apply
# must be outside class
SIGNATURE = "(f4, f4, f4, f4, f4[:], f4[:,:], f4[:], f4[:])"
if FTYPE == np.float64:
    SIGNATURE = SIGNATURE.replace("f4","f8")

@guvectorize([SIGNATURE], '(),(),(),(),(b),(b,c),(c)->(b)', target=TARGET)
def apply_sys_vectorized(
    true_energy,
    true_coszen,
    delta_index,
    energy_pivot,
    nu_flux_nominal,
    gradients,
    gradient_params,
    out,
):
    apply_sys_kernel(
        true_energy=true_energy,
        true_coszen=true_coszen,
        delta_index=delta_index,
        energy_pivot=energy_pivot,
        nu_flux_nominal=nu_flux_nominal,
        gradients=gradients,
        gradient_params=gradient_params,
        out=out,
    )
