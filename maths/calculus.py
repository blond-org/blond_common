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


def deriv_diff(x, y):
    '''
    Function returning the derivative of a given function y, using diff
    Since the returned array is 1 element shorter, a new_x array is also
    returned
    '''

    yprime = np.diff(y)/np.diff(x)
    new_x = x[0:-1]+np.diff(x)/2

    return new_x, yprime


def deriv_gradient(x, y):
    '''
    Function returning the derivative of a given function y, using gradient
    '''

    yprime = np.gradient(y, x)

    return x, yprime


def deriv_cubic(x, y, tck=None, s=0):
    '''
    Function returning the derivative of a given function y, using cubic spline
    interpolation
    '''

    if tck is None:
        tck = interp.splrep(x, y, s=s)

    yprime = interp.splev(x, tck, der=1)

    return x, yprime


def integ_trapz(x, y, constant=0.):
    '''
    Function returning the primitive of a given function y, using cumtrapz
    '''

    Y = integ.cumtrapz(y, x=x, initial=0) + constant

    return x, Y


def integ_cubic(x, y, constant=0., s=0, tck=None,
                tck_ader=None, rettck=False):
    '''
    Function returning the primitive of a given function y, using cubic spline
    interpolation
    '''

    if tck is None:
        tck = interp.splrep(x, y, s=s)
    if tck_ader is None:
        tck_ader = interp.splantider(tck)

    Y = interp.splev(x, tck_ader) + constant

    if rettck:
        return x, Y, tck, tck_ader
    else:
        return x, Y


def find_zeros_cubic(x, y, der, tck):

    pass


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

    # Taking the second derivative to find whether min or max
    sign_root = np.sign(interp.splev(roots, tck_der, der=1))

    min_pos = []
    max_pos = []
    min_val = []
    max_val = []

    for rootLoop in range(len(roots)):

        if sign_root[rootLoop] > 0:
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


def minmax_location_discrete(x, f):
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
