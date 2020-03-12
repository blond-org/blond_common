# coding: utf8
# Copyright 2019 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Base class for constructing buckets and dealing with single particle dynamics
:Authors: **Simon Albright**
"""

#General imports
import numpy as np
import scipy as sp
import scipy.interpolate as interp


def prep_interp_cubic(x, y, bc_type='not-a-knot', extrapolate=None):
    
    return interp.CubicSpline(x, y, bc_type=bc_type, 
                              extrapolate = extrapolate)
    