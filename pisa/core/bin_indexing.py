"""
Functions to retrieve the bin location of each elements
of an array, inside a Container, based on its specified 
output binning.

functions were adapted from translation.py

"""

from __future__ import absolute_import, print_function, division

import numpy as np
from numba import guvectorize, SmartArray, cuda

from pisa import FTYPE, TARGET
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.utils.log import logging, set_verbosity
from pisa.utils.numba_tools import myjit, WHERE
from pisa.utils import vectorizer

__all__ = [
    'lookup_indices']


# ---------- Lookup methods ---------------

def lookup_indices(sample, binning):
    """The inverse of histograming

    Paramters
    --------
    sample : list of SmartArrays


    binning : PISA MultiDimBinning

    Returns: for each event the index of the histogram in which it falls into

    Notes
    -----
    this method works for 1d, 2d and 3d histogram only

    """
    assert binning.num_dims in [1,2,3], 'can only do 1d, 2d and 3d at the moment'

    bin_edges = [edges.magnitude for edges in binning.bin_edges]

    array = SmartArray(np.zeros_like(sample[0]))

    if binning.num_dims == 1:

        assert len(sample) == 1,'ERROR: binning provided has 1 dimension, but sample provided has not'
        
        lookup_index_vectorized_1d(sample[0].get(WHERE),
                                   bin_edges[0],
                                   out=array.get(WHERE))

    elif binning.num_dims == 2:

        assert len(sample)==2,'ERROR: binning provided has 2 dimensions, but sample provided has not.'

        lookup_index_vectorized_2d(
            sample[0].get(WHERE),
            sample[1].get(WHERE),
            bin_edges[0],
            bin_edges[1],
            out=array.get(WHERE),
        )

    elif binning.num_dims == 3:

        assert len(sample)==3,'ERROR: binning provided has 3 dimensions, but sample provided has not.'

        lookup_index_vectorized_3d(
            sample[0].get(WHERE),
            sample[1].get(WHERE),
            sample[2].get(WHERE),
            bin_edges[0],
            bin_edges[1],
            bin_edges[2],
            out=array.get(WHERE),
        )
        
    else:
        raise NotImplementedError()
    array.mark_changed(WHERE)
    return array

#----------------------------------------------------

@myjit
def find_index(x, bin_edges):
    """simple binary search

    direct transformations instead of search
    """

    first = 0
    last = len(bin_edges) - 1
    while first <= last:
        i = int((first + last)/2)
        if x >= bin_edges[i]:
            if (x < bin_edges[i+1]) or (x <= bin_edges[-1] and i == len(bin_edges) - 1):
                break
            else:
                first = i + 1
        else:
            last = i - 1
    return i

#-----------------------------------------------------------------------
# Numba vectorized functions

if FTYPE == np.float32:
    _SIGNATURE = ['(f4[:], f4[:], f4[:])']
else:
    _SIGNATURE = ['(f8[:], f8[:], f8[:])']

@guvectorize(_SIGNATURE, '(),(j)->()', target=TARGET)
def lookup_index_vectorized_1d(sample_x, bin_edges_x, indices):
    sample_x_ = sample_x[0]

    if (sample_x_ >= bin_edges_x[0] and sample_x_ <= bin_edges_x[-1]):
        idx = find_index(sample_x_, bin_edges_x)
        indices[0] = idx
    else:
        indices[0] = 0.

#-----------------------------------------------------------------------
# Numba vectorized functions

if FTYPE == np.float32:
    _SIGNATURE = ['(f4[:], f4[:], f4[:], f4[:], f4[:])']
else:
    _SIGNATURE = ['(f8[:], f8[:], f8[:], f8[:], f8[:])']

@guvectorize(_SIGNATURE, '(),(),(j),(k)->()', target=TARGET)
def lookup_index_vectorized_2d(sample_x, sample_y, bin_edges_x, bin_edges_y, indices):
    """Same as above, except we get back the index"""
    sample_x_ = sample_x[0]
    sample_y_ = sample_y[0]
    if (sample_x_ >= bin_edges_x[0]
            and sample_x_ <= bin_edges_x[-1]
            and sample_y_ >= bin_edges_y[0]
            and sample_y_ <= bin_edges_y[-1]):
        idx_x = find_index(sample_x_, bin_edges_x)
        idx_y = find_index(sample_y_, bin_edges_y)
        idx = idx_x*(len(bin_edges_y)-1) + idx_y
        indices[0] = idx
    else:
        indices[0] = 0.




if FTYPE == np.float32:
    _SIGNATURE = ['(f4[:], f4[:], f4[:], f4[:], f4[:], f4[:], f4[:])']
else:
    _SIGNATURE = ['(f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8[:])']

@guvectorize(_SIGNATURE, '(),(),(),(j),(k),(l)->()', target=TARGET)
def lookup_index_vectorized_3d(sample_x, sample_y, sample_z,  bin_edges_x, bin_edges_y, bin_edges_z, indices):
    """Vectorized gufunc to perform the lookup"""
    sample_x_ = sample_x[0]
    sample_y_ = sample_y[0]
    sample_z_ = sample_z[0]
    if (sample_x_ >= bin_edges_x[0]
            and sample_x_ <= bin_edges_x[-1]
            and sample_y_ >= bin_edges_y[0]
            and sample_y_ <= bin_edges_y[-1]
            and sample_z_ >= bin_edges_z[0]
            and sample_z_ <= bin_edges_z[-1]):
        idx_x = find_index(sample_x_, bin_edges_x)
        idx_y = find_index(sample_y_, bin_edges_y)
        idx_z = find_index(sample_z_, bin_edges_z)
        idx = (idx_x*(len(bin_edges_y)-1) + idx_y)*(len(bin_edges_z)-1) + idx_z
        indices[0] = idx
    else:
        indices[0] = 0.






def test_histogram():
    """Unit tests for `histogram` function"""
    n_evts = 100
    x = np.array([1,1,1,1,1,3,4,5,6,7], dtype=FTYPE)
    y = np.array([2,2,2,2,2,3,3,3,3,3], dtype=FTYPE)
    z = np.array([0,0,0,0,1,0,0,0,0,0], dtype=FTYPE)

    w = np.ones(n_evts, dtype=FTYPE)

    x = SmartArray(x)
    y = SmartArray(y)
    z = SmartArray(z)

    w = SmartArray(w)

    binning_x = OneDimBinning(name='x', num_bins=7, is_lin=True, domain=[0, 7])
    binning_y = OneDimBinning(name='y', num_bins=3, is_lin=True, domain=[0, 4])
    binning_z = OneDimBinning(name='z', num_bins=2, is_lin=True, domain=[0, 1])
    
    binning_1d = MultiDimBinning([binning_x])
    binning_2d = MultiDimBinning([binning_x, binning_y])
    binning_3d = MultiDimBinning([binning_x, binning_y, binning_z])


    # 1D case: check that each event falls into its predicted bin
    print('TEST 1D:')
    print('array in 1D: ',x.get(WHERE),'\nBinning: ',binning_1d.bin_edges)
    indices = lookup_indices([x], binning_1d)
    print('indices of each array element:',indices.get(WHERE))
    print('*********************************\n')

    # 2D case:
    print('TEST 2D:')
    print('array in 2D: ',[(i,j) for i,j in zip(x.get(WHERE),y.get(WHERE))],'\nBinning: ',binning_2d.bin_edges)
    indices = lookup_indices([x,y], binning_2d)
    print('indices of each array element:',indices.get(WHERE))
    print('*********************************\n')

    # 3D case:
    print('TEST 3D:')
    print('array in 3D: ',[(i,j,k) for i,j,k in zip(x.get(WHERE),y.get(WHERE),z.get(WHERE))],'\nBinning: ',binning_3d.bin_edges)
    indices = lookup_indices([x,y,z], binning_3d)
    print('indices of each array element:',indices.get(WHERE))
    print('*********************************\n')

    logging.info('<< PASS : test_histogram >>')

if __name__ == '__main__':
    set_verbosity(1)
    test_histogram()
