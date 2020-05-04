'''
Created on 30 avr. 2020

@author: Alexandre
'''

# General import
import sys

# BLonD Common import
sys.path.append('./../../../')
from blond_common.interfaces.input_parameters.ring_section import \
    Section, machine_program


# To declare a Section with simple input
section_length = 300
alpha_0 = 1e-3
momentum = 26e9

section = Section(section_length, alpha_0, momentum)

print(section.section_length,
      section.synchronous_data,
      section.alpha_0)

# To declare a Section using other type of synchronous data
# Note that to pass bending_field, bending_radius should be
# passed as well
section_length = 300
alpha_0 = 1e-3
energy = 26e9

section = Section(section_length, alpha_0, energy=energy)

print(section.section_length,
      section.synchronous_data,
      section.alpha_0)

# Turn-by-turn input
section_length = 300
alpha_0 = [1e-3, 1e-3, 1e-3]
momentum = [26e9, 27e9, 28e9]

section = Section(section_length, alpha_0, momentum)

print(section.section_length,
      section.synchronous_data,
      section.alpha_0)

# Time based input
section_length = 300
alpha_0 = 1e-3
momentum = [[0, 1, 2],
            [26e9, 27e9, 28e9]]

section = Section(section_length, alpha_0, momentum)

print(section.section_length,
      section.synchronous_data,
      section.alpha_0)

# Using the input program function, turn based
section_length = 300
alpha_0 = 1e-3
momentum = machine_program(26e9, n_turns=5)

section = Section(section_length, alpha_0, momentum)

print(section.section_length,
      section.synchronous_data,
      section.alpha_0)

# Using the input program function, time based
section_length = 300
alpha_0 = 1e-3
momentum = machine_program([[0, 1, 2],
                            [26e9, 27e9, 28e9]],
                           interpolation='linear')

section = Section(section_length, alpha_0, momentum)

print(section.section_length,
      section.synchronous_data,
      section.alpha_0)

# Passing non-linear momentum compaction factors
section_length = 300
alpha_0 = 1e-3
alpha_1 = 1e-4
alpha_2 = 1e-5
alpha_5 = 1e-9
momentum = [26e9, 26e9, 26e9]

section = Section(section_length, alpha_0, momentum,
                  alpha_1=alpha_1, alpha_2=alpha_2, alpha_5=alpha_5)

print(section.section_length,
      section.synchronous_data,
      section.alpha_0,
      section.alpha_1,
      section.alpha_2,
      section.alpha_5,
      section.alpha_orders_defined)
