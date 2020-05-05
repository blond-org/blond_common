# coding: utf8
# Copyright 2014-2020 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Examples of usage of the Ring object.**
    :Authors: **Simon Albright**, **Alexandre Lasheen**
'''

# General import
import sys

# BLonD Common import
sys.path.append('./../../../')
from blond_common.interfaces.input_parameters.ring import \
    Ring, Section, machine_program


# To declare a Section with simple input
circumference = 628
alpha_0 = 1e-3
momentum = 26e9

ring = Ring(circumference, alpha_0, 'proton', momentum)
ring.eta_generation()

print(ring.ring_circumference,
      ring.momentum,
      ring.alpha_0)
