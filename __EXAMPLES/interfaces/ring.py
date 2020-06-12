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
from blond_common.interfaces.machine_parameters.ring import \
    Ring, RingSection, machine_program
from blond_common.interfaces.beam.beam import Proton, Electron, Particle

# To declare a Ring with simple input
length = 628
alpha_0 = 1e-3
momentum = 26e9
particle = Proton()  # can also be 'proton'

section = RingSection(length, alpha_0, momentum)
ring = Ring(particle, section)

# To declare a Ring with other particle type (electrons)
length = 80e3
alpha_0 = 1e-3
momentum = 120e9
particle = Electron()  # can also be 'electron'

section = RingSection(length, alpha_0, momentum)
ring = Ring(particle, section)

# To declare a Ring with other particle type (ions)
length = 80e3
alpha_0 = 1e-3
momentum = 120e9
user_mass = 208 * Proton().mass
user_charge = 82
particle = Particle(user_mass, user_charge)

section = RingSection(length, alpha_0, momentum)
ring = Ring(particle, section)

# To declare a Ring with multiple sections with same parameters
length = 628
alpha_0 = 1e-3
momentum = 26e9
particle = Proton()

section_1 = RingSection(length, alpha_0, momentum)
section_2 = RingSection(length, alpha_0, momentum)
section_3 = RingSection(length, alpha_0, momentum)
ring = Ring(particle, [section_1, section_2, section_3])

# To declare a Ring with two simple sections
length = 628 / 2
alpha_0 = 1e-3
momentum = [26e9, 26e9, 26e9]
orbit_bump = 1e-3
particle = Proton()  # can also be 'proton'

section_1 = RingSection(length, alpha_0, momentum)
section_2 = RingSection(length, alpha_0, momentum, orbit_bump=orbit_bump)
ring = Ring(particle, [section_1, section_2])

print(ring.circumference_design,
      ring.circumference,
      ring.section_length,
      ring.momentum,
      ring.t_rev_design,
      ring.t_rev)
