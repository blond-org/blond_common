# coding: utf8
# Copyright 2019 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Unit-test for impedance_sources.py
:Authors: **Markus Schwarz**
"""

# General imports
# ---------------
import sys
import os
import unittest
import numpy as np

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

# BLonD_Common imports
# --------------------
if os.path.abspath(this_directory + '../../../../../') not in sys.path:
    sys.path.insert(0, os.path.abspath(this_directory + '../../../../../'))

import blond_common.interfaces.impedances.impedance_sources as impSource

class TestDistributionsBaseClass(unittest.TestCase):
    
    def test_base(self):
        pass


    def test_input_table(self):
        
        
        
    
if __name__ == '__main__':

    unittest.main()