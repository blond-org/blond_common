# coding: utf8
# Copyright 2019 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Unit-test for the blond_common.fitting.profile module

:Authors: **Alexandre Lasheen**

"""

# General imports
# ---------------
import sys
import unittest
import numpy as np
import os

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

# fitting.profile tests imports
# -----------------------------
if os.path.abspath(this_directory + '../../../') not in sys.path:
    sys.path.insert(0, os.path.abspath(this_directory + '../../../'))
# sys.path.append('./../../../')

from blond_common.fitting.profile import RMS


class TestFittingProfile(unittest.TestCase):

    # Initialization ----------------------------------------------------------

    def setUp(self):

        self.test = 0

    def test_1(self):

        pass


if __name__ == '__main__':

    unittest.main()
