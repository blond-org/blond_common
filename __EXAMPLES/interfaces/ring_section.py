# coding: utf8
# Copyright 2014-2020 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Examples of usage of the RingSection object.**
    :Authors: **Simon Albright**, **Alexandre Lasheen**
'''

# General import
import sys

# BLonD Common import
sys.path.append('./../../../')
from blond_common.interfaces.input_parameters.ring_section import \
    RingSection, machine_program


# To declare a RingSection with simple input
length = 300
alpha_0 = 1e-3
momentum = 26e9

section = RingSection(length, alpha_0, momentum)

print(section.length,
      section.synchronous_data,
      section.alpha_0)

# To declare a RingSection using other type of synchronous data
# Note that to pass bending_field, bending_radius should be
# passed as well
length = 300
alpha_0 = 1e-3
energy = 26e9

section = RingSection(length, alpha_0, energy=energy)

print(section.length,
      section.synchronous_data,
      section.alpha_0)

# To declare a RingSection with turn-by-turn program as input
length = 300
alpha_0 = [1e-3, 1e-3, 1e-3]
momentum = [26e9, 27e9, 28e9]

section = RingSection(length, alpha_0, momentum)

print(section.length,
      section.synchronous_data,
      section.alpha_0)

# To declare a RingSection with time based program as input
length = 300
alpha_0 = 1e-3
momentum = [[0, 1, 2],
            [26e9, 27e9, 28e9]]

section = RingSection(length, alpha_0, momentum)

print(section.length,
      section.synchronous_data,
      section.alpha_0)

# To declare a RingSection using the input program function, turn based
length = 300
alpha_0 = 1e-3
momentum = machine_program(26e9, n_turns=5)

section = RingSection(length, alpha_0, momentum)

print(section.length,
      section.synchronous_data,
      section.alpha_0)

# To declare a RingSection using the input program function, time based
length = 300
alpha_0 = 1e-3
momentum = machine_program([[0, 1, 2],
                            [26e9, 27e9, 28e9]],
                           interpolation='linear')

section = RingSection(length, alpha_0, momentum)

print(section.length,
      section.synchronous_data,
      section.alpha_0)

# To declare a RingSection passing non-linear momentum compaction factors
length = 300
alpha_0 = 1e-3
alpha_1 = 1e-4
alpha_2 = 1e-5
alpha_5 = 1e-9
energy = machine_program([[0, 1, 2],
                          [26e9, 27e9, 28e9]])

section = RingSection(length, alpha_0, energy=energy,
                      alpha_1=alpha_1, alpha_2=alpha_2, alpha_5=alpha_5)

print(section.length,
      section.synchronous_data,
      section.alpha_0,
      section.alpha_1,
      section.alpha_2,
      section.alpha_5,
      section.alpha_orders_defined)
