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
    Ring, RingSection, machine_program

# To declare a Ring with simple input
length = 628
alpha_0 = 1e-3
momentum = 26e9

section = RingSection(length, alpha_0, momentum)
ring = Ring('proton', section)

# To declare a Ring with two simple sections
length = 628 / 2
alpha_0 = 1e-3
momentum = [26e9, 26e9, 26e9]
orbit_length = length + 1

section_1 = RingSection(length, alpha_0, momentum)
section_2 = RingSection(length, alpha_0, momentum, orbit_length=orbit_length)
ring = Ring('proton', [section_1, section_2])

print(ring.circumference, ring.section_length, ring.orbit_length,
      ring.momentum, ring.t_rev)



