# pylint: disable=redefined-outer-name, missing-docstring


"""
Collection of useful vectorized functions
"""

from __future__ import absolute_import, print_function

import math

import numpy as np
from numba import guvectorize, SmartArray

from pisa import FTYPE, TARGET
from pisa.utils.log import logging, set_verbosity
from pisa.utils.numba_tools import WHERE


__all__ = [
    "mul",
    "imul",
    "imul_and_scale",
    "itruediv",
    "assign",
    "pow",
    "sqrt",
    "replace_where_counts_gt",
]

__version__ = "0.2"
__author__ = "Philipp Eller (pde3@psu.edu)"


FX = "f4" if FTYPE == np.float32 else "f8"


# ---------------------------------------------------------------------------- #


def scale(vals, scale, out):
    """Multiply .. ::

        out[:] = vals[:] * scale

    """
    scale_gufunc(vals.get(WHERE), FTYPE(scale), out=out.get(WHERE))
    out.mark_changed(WHERE)


@guvectorize([f"({FX}[:], {FX}, {FX}[:])"], "(), () -> ()", target=TARGET)
def scale_gufunc(vals, scale, out):
    out[0] = vals[0] * scale


# ---------------------------------------------------------------------------- #


def mul(vals0, vals1, out):
    """Multiply .. ::

        out[:] = vals0[:] * vals1[:]

    """
    mul_gufunc(vals0.get(WHERE), vals1.get(WHERE), out=out.get(WHERE))
    out.mark_changed(WHERE)


@guvectorize([f"({FX}[:], {FX}[:], {FX}[:])"], "(), () -> ()", target=TARGET)
def mul_gufunc(vals0, vals1, out):
    out[0] = vals0[0] * vals1[0]


# ---------------------------------------------------------------------------- #


def imul(vals, out):
    """Multiply augmented assignment of two arrays .. ::

        out[:] *= vals[:]

    """
    imul_gufunc(vals.get(WHERE), out=out.get(WHERE))
    out.mark_changed(WHERE)


@guvectorize([f"({FX}[:], {FX}[:])"], "() -> ()", target=TARGET)
def imul_gufunc(vals, out):
    out[0] *= vals[0]


# ---------------------------------------------------------------------------- #


def imul_and_scale(vals, scale, out):
    """Multiply and scale augmented assignment .. ::

        out[:] *= vals[:] * scale

    """
    imul_and_scale_gufunc(vals.get(WHERE), FTYPE(scale), out=out.get(WHERE))
    out.mark_changed(WHERE)


@guvectorize([f"({FX}[:], {FX}, {FX}[:])"], "(), () -> ()", target=TARGET)
def imul_and_scale_gufunc(vals, scale, out):
    out[0] *= vals[0] * scale


def test_imul_and_scale():
    """Unit tests for function ``imul_and_scale``"""
    a = SmartArray(np.linspace(0, 1, 1000, dtype=FTYPE))
    out = SmartArray(np.ones_like(a))
    imul_and_scale(vals=a, scale=10.0, out=out)
    assert np.allclose(out.get("host"), np.linspace(0, 10, 1000, dtype=FTYPE))
    logging.info("<< PASS : test_multiply_and_scale >>")


# ---------------------------------------------------------------------------- #


def itruediv(vals, out):
    """Divide augmented assignment .. ::

        out[:] /= vals[:]

    Division by zero results in 0 for that element.
    """
    itruediv_gufunc(vals.get(WHERE), out=out.get(WHERE))
    out.mark_changed(WHERE)


@guvectorize([f"({FX}[:], {FX}[:])"], "() -> ()", target=TARGET)
def itruediv_gufunc(vals, out):
    if vals[0] == 0.0:
        out[0] = 0.0
    else:
        out[0] /= vals[0]


# ---------------------------------------------------------------------------- #


def assign(vals, out):  # pylint: disable=redefined-builtin
    """Assign array vals from another array .. ::

        out[:] = vals[:]

    """
    assign_gufunc(vals.get(WHERE), out=out.get(WHERE))
    out.mark_changed(WHERE)


@guvectorize([f"({FX}[:], {FX}[:])"], "() -> ()", target=TARGET)
def assign_gufunc(vals, out):
    out[0] = vals[0]


# ---------------------------------------------------------------------------- #


def pow(vals, pwr, out):  # pylint: disable=redefined-builtin
    """Raise vals to pwr.. ::

        out[:] = vals[:]**pwr

    """
    pow_gufunc(vals.get(WHERE), FTYPE(pwr), out=out.get(WHERE))
    out.mark_changed(WHERE)


@guvectorize([f"({FX}[:], {FX}, {FX}[:])"], "(), () -> ()", target=TARGET)
def pow_gufunc(vals, pwr, out):
    out[0] = vals[0] ** pwr


# ---------------------------------------------------------------------------- #


def sqrt(vals, out):
    """Square root of vals .. ::

        out[:] = sqrt(vals[:])

    """
    sqrt_gufunc(vals.get(WHERE), out=out.get(WHERE))
    out.mark_changed(WHERE)


@guvectorize([f"({FX}[:], {FX}[:])"], "() -> ()", target=TARGET)
def sqrt_gufunc(vals, out):
    out[0] = math.sqrt(vals[0])


# ---------------------------------------------------------------------------- #


def replace_where_counts_gt(vals, counts, min_count, out):
    """Replace `out[i]` with `vals[i]` where `counts[i]` > `min_count`"""
    replace_where_counts_gt_gufunc(
        vals.get(WHERE), counts.get(WHERE), FTYPE(min_count), out=out.get(WHERE),
    )


@guvectorize([f"({FX}[:], {FX}[:], {FX}, {FX}[:])"], "(), (), () -> ()", target=TARGET)
def replace_where_counts_gt_gufunc(vals, counts, min_count, out):
    """Replace `out[i]` with `vals[i]` where `counts[i]` > `min_count`"""
    if counts[0] > min_count:
        out[0] = vals[0]


# ---------------------------------------------------------------------------- #


if __name__ == "__main__":
    set_verbosity(1)
    test_imul_and_scale()
