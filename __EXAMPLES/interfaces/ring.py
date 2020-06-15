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

ring = Ring(particle, RingSection(length, alpha_0, momentum))

# To declare a Ring with other synchronous data (all possible definitions
# from the RingSection object)
length = 628
alpha_0 = 1e-3
tot_energy = 26e9
particle = Proton()  # can also be 'proton'

section = RingSection(length, alpha_0, energy=tot_energy)
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

# To declare a Ring with non-linear momentum compaction factor
length = 628
alpha_0 = 1e-3
alpha_1 = 1e-6
momentum = 26e9
particle = Proton()

section = RingSection(length, alpha_0, momentum, alpha_1=alpha_1)
ring = Ring(particle, section)

print(ring.alpha_1)

# To declare a Ring with a ramp (turn based program)
length = 628
alpha_0 = 1e-3
momentum = [26e9, 27e9, 28e9]
particle = Proton()

section = RingSection(length, alpha_0, momentum)
ring = Ring(particle, section)

# To declare a Ring with a ramp (time based program)
length = 628
alpha_0 = 1e-3
momentum = [[0, 0.1, 0.2], [26e9, 27e9, 28e9]]
particle = Proton()

section = RingSection(length, alpha_0, momentum)
ring = Ring(particle, section)

# To declare a Ring with time based programs
length = 628
alpha_0 = [[0, 0.1, 0.2], [1e-3, 1.1e-3, 1e-3]]
alpha_1 = [[0, 0.1, 0.2], [1e-6, 0.9e-6, 1e-6]]
momentum = [[0, 0.1, 0.2], [26e9, 26e9, 26e9]]
orbit_bump = [[0, 0.1, 0.2], [0., 1e-3, 0.]]
particle = Proton()

section = RingSection(length, alpha_0, momentum, alpha_1=alpha_1,
                      orbit_bump=orbit_bump)
ring = Ring(particle, section, eta_orders=1)

# To declare a Ring with multiple sections (parameters can be adjusted
# section by section)
length = 628 / 3
alpha_0 = 1e-3
momentum = [26e9, 26e9, 26e9]
particle = Proton()

section_1 = RingSection(length, alpha_0, momentum)
section_2 = RingSection(length, alpha_0, momentum)
section_3 = RingSection(length, alpha_0, momentum)
ring = Ring(particle, [section_1, section_2, section_3])

print(ring.circumference_design,
      ring.circumference,
      ring.section_length,
      ring.momentum,
      ring.t_rev_design,
      ring.t_rev)

# To declare a Ring with multiple sections and different momentum program
length = 628 / 3
alpha_0 = 1e-3
momentum_1 = [26e9, 27e9, 28e9]
momentum_2 = [26.5e9, 27.5e9, 28.5e9]
particle = Proton()

section_1 = RingSection(length, alpha_0, momentum_1)
section_2 = RingSection(length, alpha_0, momentum_2)
ring = Ring(particle, [section_1, section_2])

print(ring.circumference_design,
      ring.circumference,
      ring.section_length,
      ring.momentum,
      ring.t_rev_design,
      ring.t_rev)
