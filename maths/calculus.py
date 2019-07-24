# coding: utf8
# Copyright 2014-2019 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Module to perform numeral calculus: integrals, derivatives, find 0, minima,
maxima**

:Authors: **Alexandre Lasheen**
'''


# External imports
import numpy as np
import scipy.interpolate as interp
import scipy.integrate as integ
import warnings


def integ_linear(x, y, constant=0.):
    '''
    Function returning the integral of a given function y, using cumtrapz
    '''

    out = integ.cumtrapz(y, x=x, initial=0) + constant

    return out


def integ_cubic(x, y, constant=0., s=0, tck=None):
    '''
    Function returning the integral of a given function y, using cubic spline
    interpolation
    '''

    if tck is None:
        tck = interp.splrep(x, y, s=s)

    x = np.atleast_1d(x)
    dx = x[1]-x[0]
    out = np.zeros(x.shape, dtype=y.dtype)

    for n in range(len(out)):
        out[n] = interp.splint(0, x[n], tck)
    out += constant
    out *= dx

    return out


def deriv_cubic(x, y, tck=None, s=0):
    '''
    Function returning the derivative of a given function y, using cubic spline
    interpolation
    '''

    if tck is None:
        tck = interp.splrep(x, y, s=s)
    x = np.atleast_1d(x)
    out = interp.splev(x, tck, der=1)

    return out


def minmax_location_cubic(x, y, der=None, tck=None,
                          tck_der=None, s=0, rettck=False,
                          mest=10):
    '''
    Function returning the minima, maxima of a given function y,
    as well as their location in x
    '''

    if tck is None:
        tck = interp.splrep(x, y, s=s)
    if tck_der is None and der is None:
        der = deriv_cubic(y, x, tck, s=s)
    if tck_der is None:
        tck_der = interp.splrep(x, der, s=s)

    roots = interp.sproot(tck_der, mest=mest)

    if len(roots) == 0:
        return None

    values = interp.splev(roots, tck)

    min_pos = []
    max_pos = []
    min_val = []
    max_val = []

    index_slope = 1/(x[1]-x[0])
    index_origin = -index_slope*x[0]

    for rootLoop in range(len(roots)):
        index_root = roots[rootLoop] * index_slope + index_origin
        sign_root = np.sign(
            interp.splev(
                x[int(np.ceil(index_root))], tck_der) -
            interp.splev(x[int(np.floor(index_root))], tck_der))

        if sign_root > 0:
            min_pos.append(roots[rootLoop])
            min_val.append(values[rootLoop])
        else:
            max_pos.append(roots[rootLoop])
            max_val.append(values[rootLoop])

    if rettck:
        return ([np.array(min_pos), np.array(max_pos)],
                [np.array(min_val), np.array(max_val)],
                tck)
    else:
        return ([np.array(min_pos), np.array(max_pos)],
                [np.array(min_val), np.array(max_val)])


def minmax_location(x, f):
    '''Function to locate the minima and maxima of the f(x)
    numerical function.'''

    f_derivative = np.diff(f)
    x_derivative = x[0:-1] + (x[1]-x[0])/2
    f_derivative = np.interp(x, x_derivative, f_derivative)

    f_derivative_second = np.diff(f_derivative)
    f_derivative_second = np.interp(x, x_derivative,
                                    f_derivative_second)

    warnings.filterwarnings("ignore")

    f_derivative_zeros = np.unique(
        np.append(np.where(f_derivative == 0),
                  np.where(f_derivative[1:]/f_derivative[0:-1] < 0)))
    min_x_position = (
        x[f_derivative_zeros[f_derivative_second[f_derivative_zeros] > 0] + 1]
        + x[f_derivative_zeros[f_derivative_second[f_derivative_zeros] > 0]])/2
    max_x_position = (
        x[f_derivative_zeros[f_derivative_second[f_derivative_zeros] < 0] + 1]
        + x[f_derivative_zeros[f_derivative_second[f_derivative_zeros] < 0]])/2

    min_values = np.interp(min_x_position, x, f)
    max_values = np.interp(max_x_position, x, f)

    warnings.filterwarnings("default")

    return [min_x_position, max_x_position], [min_values, max_values]
