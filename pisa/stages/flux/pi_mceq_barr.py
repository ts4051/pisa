
"""
Stage to implement the intrimsic neutrino flux as calculated with MCEq, 
and the systematic flux variations based on the Barr scheme. 

It requires spline tables created by the `$PISA/scripts/create_barr_sys_tables_mceq.py`
"""
from __future__ import absolute_import, print_function, division

import math
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

    table_file : str
        pointing to spline table obtained from MCEq
    barr_* : quantity (dimensionless)

    Notes
    -----
    The table consists of 2 solutions of the cascade equation per Barr variable (12) 
    - one solution for meson and one solution for the antimeson. 
    Each solution consists of 8 splines: idx=0,2,4,6=numu, numubar, nue, nuebar. 
    idx=1,3,5,7=gradients of numu, numubar, nue, nuebar. 

    """

    def __init__(self,
                 data=None,
                 params=None,
                 input_names=None,
                 output_names=None,
                 debug_mode=None,
                 input_specs=None,
                 calc_specs=None,
                 output_specs=None,
                ):

        expected_params = ('table_file',
                           'barr_a',
                           'barr_b',
                           'barr_c',
                           'barr_d',
                           'barr_e',
                           'barr_f',
                           'barr_g',
                           'barr_h',
                           'barr_i',
                           'barr_x',
                           'barr_w',
                           'barr_y',
                           'barr_z',
                           'delta_index',
                          )
        input_names = ()
        output_names = ()

        # what are the keys used from the inputs during apply
        input_calc_keys = (
                         )
        # what are keys added or altered in the calculation used during apply
        output_calc_keys = (
                            'nominal_nu_flux',
                            'nominal_nubar_flux',
                            'sys_flux',
                           )
        # what keys are added or altered for the outputs during apply
        output_apply_keys = (
                            'nominal_nu_flux',
                            'nominal_nubar_flux',
                            'sys_flux',
                            )

        # init base class
        super(pi_mceq_barr, self).__init__(data=data,
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

	print('----- The MCEq stage has been initialized -----')

    def setup_function(self):

      # load MCeq tables
      spline_tables_dict = pickle.load(BZ2File(find_resource(self.params.table_file.value)))
      print('---- The spline tables have been loaded ----')
      #print(spline_tables_dict.keys())
      self.data.data_specs = self.calc_specs

      if self.calc_mode == 'binned':
        # speed up calculation by adding links
        # as layers don't care about flavour
        self.data.link_containers('nu', ['nue_cc', 'numu_cc', 'nutau_cc',
                                             'nue_nc', 'numu_nc', 'nutau_nc',
                                             'nuebar_cc', 'numubar_cc', 'nutaubar_cc',
                                             'nuebar_nc', 'numubar_nc', 'nutaubar_nc'])
      print('---- The containers have been linked ----')

      for container in self.data:
        container['sys_flux'] = np.empty((container.size, 2), dtype=FTYPE)
        container['nominal_nu_flux'] = np.empty((container.size, 2), dtype=FTYPE)
        container['nominal_nubar_flux'] = np.empty((container.size, 2), dtype=FTYPE)
        # evaluate the splines (flux and deltas) for each E/CZ point
        # at the moment this is done on CPU, therefore we force 'host'
        for key in spline_tables_dict.keys():
          logging.info('Evaluating MCEq splines for %s for Barr parameter %s'%(container.name, key))
          #print('Evaluating MCEq splines for %s for Barr parameter %s'%(container.name, key))
          container['barr_'+key] = np.empty((container.size, 8), dtype=FTYPE)
          #print(container['true_energy'].get('host'), container['true_coszen'].get('host'))
          self.eval_spline(container['true_energy'].get('host'),
                                 container['true_coszen'].get('host'),
                                 spline_tables_dict[key],
                                 out=container['barr_'+key].get('host'))
          container['barr_'+key].mark_changed('host')
      print('---- The splines have been evaluated ----')
      self.data.unlink_containers()


    def eval_spline(self, true_energy, true_coszen, splines, out):
        '''
        dumb function to iterate trhouh all E, CZ values
        and evlauate all 8 Barr splines at these points
        '''
        for i in xrange(len(true_energy)):

            abs_cos = abs(true_coszen[i])
            log_e = np.log(true_energy[i])
            for j in xrange(len(splines)):
                out[i,j] = splines[j](abs_cos, log_e)[0,0]


    @profile
    def compute_function(self):

      self.data.data_specs = self.calc_specs

      # The nominal flux from MCEq splines
      # Since all nominal sploines are equal, coose a random one - like c
      for container in self.data:
      	container['nominal_nu_flux'][:,0] = container['barr_c+'].get('host')[:,4]*1e4
      	container['nominal_nu_flux'][:,1] = container['barr_c+'].get('host')[:,0]*1e4
      	container['nominal_nubar_flux'][:,0] = container['barr_c+'].get('host')[:,6]*1e4
      	container['nominal_nubar_flux'][:,1] = container['barr_c+'].get('host')[:,2]*1e4

      #barr_a = self.params.barr_a.value.m_as('dimensionless')
      #barr_b = self.params.barr_b.value.m_as('dimensionless')
      #barr_c = sself.params.barr_c.value.m_as('dimensionless')
      #barr_d = self.params.barr_d.value.m_as('dimensionless')
      #barr_e = self.params.barr_e.value.m_as('dimensionless')
      #barr_f = self.params.barr_f.value.m_as('dimensionless')
      barr_g = self.params.barr_g.value.m_as('dimensionless')
      barr_h = self.params.barr_h.value.m_as('dimensionless')
      barr_i = self.params.barr_i.value.m_as('dimensionless')
      barr_w = self.params.barr_w.value.m_as('dimensionless')
      #barr_x = self.params.barr_x.value.m_as('dimensionless')
      barr_y = self.params.barr_y.value.m_as('dimensionless')
      barr_z = self.params.barr_z.value.m_as('dimensionless')

      delta_index = self.params.delta_index.value.m_as("dimensionless")

      # Apply the Barr modifications to nominal flux 
      for container in self.data:
        apply_sys_vectorized(
          container["true_energy"].get(WHERE),
          container["true_coszen"].get(WHERE),
          container["nominal_nu_flux"].get(WHERE),
          container["nominal_nubar_flux"].get(WHERE),
          container["nubar"],
          delta_index,
          #container['barr_a+'].get(WHERE), container['barr_a-'].get(WHERE), barr_a,
          #container['barr_b+'].get(WHERE), container['barr_b-'].get(WHERE), barr_b,
          #container['barr_c+'].get(WHERE), container['barr_c-'].get(WHERE), barr_c,
          #container['barr_d+'].get(WHERE), container['barr_d-'].get(WHERE), barr_d,
          #container['barr_e+'].get(WHERE), container['barr_e-'].get(WHERE), barr_e,
          #container['barr_f+'].get(WHERE), container['barr_f-'].get(WHERE), barr_f,
          container['barr_g+'].get(WHERE), container['barr_g-'].get(WHERE), barr_g,
          container['barr_h+'].get(WHERE), container['barr_h-'].get(WHERE), barr_h,
          container['barr_i+'].get(WHERE), container['barr_i-'].get(WHERE), barr_i,
          #container['barr_x+'].get(WHERE), container['barr_x-'].get(WHERE), barr_x,
          container['barr_w+'].get(WHERE), container['barr_w-'].get(WHERE), barr_w,
          container['barr_y+'].get(WHERE), container['barr_y-'].get(WHERE), barr_y,
          container['barr_z+'].get(WHERE), container['barr_z-'].get(WHERE), barr_z,
          out=container['sys_flux'].get(WHERE),
        )
        container['sys_flux'].mark_changed(WHERE)

@myjit
def spectral_index_scale(true_energy, egy_pivot, delta_index):
    """ calculate spectral index scale """
    return math.pow((true_energy / egy_pivot), delta_index)

@myjit
def add_barr(idx,
               #barr_a_pos, barr_a_neg, barr_a,
               #barr_b_pos, barr_b_neg, barr_b,
               #barr_c_pos, barr_c_neg, barr_c,
               #barr_d_pos, barr_d_neg, barr_d,
               #barr_e_pos, barr_e_neg, barr_e,
               #barr_f_pos, barr_f_neg, barr_f,
               barr_g_pos, barr_g_neg, barr_g,
               barr_h_pos, barr_h_neg, barr_h,
               barr_i_pos, barr_i_neg, barr_i,
               barr_w_pos, barr_w_neg, barr_w,
               #barr_x_pos, barr_x_neg, barr_x,
               barr_y_pos, barr_y_neg, barr_y,
               barr_z_pos, barr_z_neg, barr_z,
               ):
  return (0
          + barr_g*(barr_g_pos[idx]+barr_g_neg[idx])*1e4
          + barr_h*(barr_h_pos[idx]+barr_h_neg[idx])*1e4
          + barr_i*(barr_i_pos[idx]+barr_i_neg[idx])*1e4
          + barr_w*(barr_w_pos[idx]+barr_w_neg[idx])*1e4
          + barr_y*(barr_y_pos[idx]+barr_y_neg[idx])*1e4
          + barr_z*(barr_z_pos[idx]+barr_z_neg[idx])*1e4
          )

@myjit
def apply_sys_kernel(
    true_energy,
    true_coszen,
    nominal_nu_flux,
    nominal_nubar_flux,
    nubar,
    delta_index,
    barr_g_pos, barr_g_neg, barr_g,
    barr_h_pos, barr_h_neg, barr_h,
    barr_i_pos, barr_i_neg, barr_i,
    barr_w_pos, barr_w_neg, barr_w,
    barr_y_pos, barr_y_neg, barr_y,
    barr_z_pos, barr_z_neg, barr_z,
    out,
):

  if nubar > 0: # If particle 
    out[0] = nominal_nu_flux[0]+add_barr(5, 
                                          barr_g_pos, barr_g_neg, barr_g,
                                          barr_h_pos, barr_h_neg, barr_h,
                                          barr_i_pos, barr_i_neg, barr_i,
                                          barr_w_pos, barr_w_neg, barr_w,
                                          barr_y_pos, barr_y_neg, barr_y,
                                          barr_z_pos, barr_z_neg, barr_z)
    out[1] = nominal_nu_flux[1]+add_barr(1, 
                                          barr_g_pos, barr_g_neg, barr_g,
                                          barr_h_pos, barr_h_neg, barr_h,
                                          barr_i_pos, barr_i_neg, barr_i,
                                          barr_w_pos, barr_w_neg, barr_w,
                                          barr_y_pos, barr_y_neg, barr_y,
                                          barr_z_pos, barr_z_neg, barr_z)
  else: # if antiparticle 
    out[0] = nominal_nubar_flux[0]+add_barr(7, 
                                          barr_g_pos, barr_g_neg, barr_g,
                                          barr_h_pos, barr_h_neg, barr_h,
                                          barr_i_pos, barr_i_neg, barr_i,
                                          barr_w_pos, barr_w_neg, barr_w,
                                          barr_y_pos, barr_y_neg, barr_y,
                                          barr_z_pos, barr_z_neg, barr_z)
    out[1] = nominal_nubar_flux[1]+add_barr(3, 
                                          barr_g_pos, barr_g_neg, barr_g,
                                          barr_h_pos, barr_h_neg, barr_h,
                                          barr_i_pos, barr_i_neg, barr_i,
                                          barr_w_pos, barr_w_neg, barr_w,
                                          barr_y_pos, barr_y_neg, barr_y,
                                          barr_z_pos, barr_z_neg, barr_z)
  print(nubar, out)

  idx_scale = spectral_index_scale(true_energy, 24.0900951261, delta_index)
  out[0] *= idx_scale
  out[1] *= idx_scale

# vectorized function to apply
# must be outsixsde class
if FTYPE == np.float64:
    #SIGNATURE = "(f8, f8, f8[:], f8[:], i4, f8, f8[:])"
    SIGNATURE = "(f8, f8,\
                  f8[:], f8[:],\
                  i4, f8,\
                  f8[:], f8[:], f8,\
                  f8[:], f8[:], f8,\
                  f8[:], f8[:], f8,\
                  f8[:], f8[:], f8,\
                  f8[:], f8[:], f8,\
                  f8[:], f8[:], f8,\
                  f8[:])"
else:
    #SIGNATURE = "(f4, f4, f4[:], f4[:], i4, f4, f4[:])"
    SIGNATURE = "(f4, f4,\
                  f4[:], f4[:],\
                  i4, f4,\
                  f4[:], f4[:], f4,\
                  f4[:], f4[:], f4,\
                  f4[:], f4[:], f4,\
                  f4[:], f4[:], f4,\
                  f4[:], f4[:], f4,\
                  f4[:], f4[:], f4,\
                  f4[:])"

#@guvectorize([SIGNATURE], "(),(),(d),(d),(),()->(d)", target=TARGET)
@guvectorize([SIGNATURE], '(),(),(d),(d),(),(),\
                           (c),(c),(),\
                           (c),(c),(),\
                           (c),(c),(),\
                           (c),(c),(),\
                           (c),(c),(),\
                           (c),(c),()\
                           ->(d)', target=TARGET)

def apply_sys_vectorized(
    true_energy,
    true_coszen,
    nominal_nu_flux,
    nominal_nubar_flux,
    nubar,
    delta_index,
    barr_g_pos, barr_g_neg, barr_g,
    barr_h_pos, barr_h_neg, barr_h,
    barr_i_pos, barr_i_neg, barr_i,
    barr_w_pos, barr_w_neg, barr_w,
    barr_y_pos, barr_y_neg, barr_y,
    barr_z_pos, barr_z_neg, barr_z,
    out,
    ):

    apply_sys_kernel(
      true_energy,
      true_coszen,
      nominal_nu_flux,
      nominal_nubar_flux,
      nubar,
      delta_index,
      barr_g_pos, barr_g_neg, barr_g,
      barr_h_pos, barr_h_neg, barr_h,
      barr_i_pos, barr_i_neg, barr_i,
      barr_w_pos, barr_w_neg, barr_w,
      barr_y_pos, barr_y_neg, barr_y,
      barr_z_pos, barr_z_neg, barr_z,
      out,
      )
