# coding: utf8
# Copyright 2014-2019 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Module with residue functions used for fitting**

:Authors: **Alexandre Lasheen**, **Markus Schwarz**
'''

# General imports
import numpy as np


def vertical_least_square(fitParameters, *fittingArgList):
    '''
    * Function to be used for fitting in the minimize function (least square).*
    '''

    profile_fit_function = fittingArgList[0]
    time_array = fittingArgList[1]
    fittedProfileInputY = fittingArgList[2]

    residue = np.sum((fittedProfileInputY -
                      profile_fit_function(time_array, *fitParameters))**2)

    return residue
