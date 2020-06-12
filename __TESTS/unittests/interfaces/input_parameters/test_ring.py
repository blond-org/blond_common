# coding: utf8
# Copyright 2014-2020 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Unit-test for blond_common.interfaces.input_parameters.ring.py
:Authors: **Markus Schwarz**, **Alexandre Lasheen**
"""

# General imports
# ---------------
import sys
import unittest
import numpy as np
import os
import warnings

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

# BLonD_Common imports
# --------------------
if os.path.abspath(this_directory + '../../../../../') not in sys.path:
    sys.path.insert(0, os.path.abspath(this_directory + '../../../../../'))

from blond_common.interfaces.input_parameters.ring import Ring, RingSection, \
    machine_program
from blond_common.interfaces.beam.beam import Proton
from blond_common.devtools import exceptions as excpt
from blond_common import datatypes as dTypes


class TestRing(unittest.TestCase):

    # Initialization ----------------------------------------------------------

    def setUp(self):

        pass

    def assertIsNaN(self, value, msg=None):
        """
        Fail if provided value is not NaN
        """

        standardMsg = "%s is not NaN" % str(value)

        if not np.isnan(value):
            self.fail(self._formatMessage(msg, standardMsg))

    # Input test --------------------------------------------------------------

#     def test_simple_input(self):
#         # Test the simplest input
# 
#         length = 300  # m
#         alpha_0 = 1e-3
#         momentum = 26e9  # eV
#         particle = Proton()
# 
#         section = RingSection(length, alpha_0, momentum)
#         ring = Ring(particle, section)
# 
#         with self.subTest('Simple input - length'):
#             np.testing.assert_equal(
#                 length, ring.circumference)

    # Exception raising test --------------------------------------------------


if __name__ == '__main__':

    unittest.main()
