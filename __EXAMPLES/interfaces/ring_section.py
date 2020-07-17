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
    :Authors: **Alexandre Lasheen**
'''

# BLonD Common import
# Add blond_common to PYTHONPATH if not done
from blond_common.interfaces.machine_parameters.ring_section import \
    RingSection, machine_program


# To declare a RingSection with simple input
length = 300  # m
alpha_0 = 1e-3
momentum = 26e9  # eV/c

section = RingSection(length, alpha_0, momentum)

print('-- Simple input')
print(f'Section length {section.length} [m]')
print(f'Sync. data - Momentum {section.synchronous_data} [eV/c]')
print(f'Linear momentum compaction {section.alpha_0}')
print()


# To declare a RingSection using other type of synchronous data
# Note that to pass bending_field, bending_radius should be
# passed as well
length = 300  # m
alpha_0 = 1e-3
energy = 26e9  # eV

section = RingSection(length, alpha_0, energy=energy)

print('-- Other sync data - Total energy')
print(f'Section length {section.length} [m]')
print(f'Sync. data - Total energy {section.synchronous_data} [eV/c]')
print(f'Linear momentum compaction {section.alpha_0}')
print()


# To declare a RingSection using the dipole bending field and bending radius
length = 300  # m
alpha_0 = 1e-3
bending_field = 1.  # T
bending_radius = 749  # m

section = RingSection(length, alpha_0,
                      bending_field=bending_field,
                      bending_radius=bending_radius)

print('-- Other sync. data - B field')
print(f'Section length {section.length} [m]')
print(f'Sync. data - Momentum {section.synchronous_data} [eV/c]')
print(f'Linear momentum compaction {section.alpha_0}')
print()


# To declare a RingSection with turn-by-turn program as input
length = 300  # m
alpha_0 = [1e-3, 1e-3, 1e-3]
momentum = [26e9, 27e9, 28e9]  # eV/c

section = RingSection(length, alpha_0, momentum)

print('-- Programs - Turn by turn')
print(f'Section length {section.length} [m]')
print(f'Sync. data - Momentum {section.synchronous_data} [eV/c]')
print(f'Linear momentum compaction {section.alpha_0}')
print()


# To declare a RingSection using the input program function, turn based
length = 300  # m
alpha_0 = 1e-3
momentum = machine_program(26e9, n_turns=5)  # eV/c

section = RingSection(length, alpha_0, momentum)

print('-- Programs - Turn based with machine_program')
print(f'Section length {section.length} [m]')
print(f'Sync. data - Momentum {section.synchronous_data} [eV/c]')
print(f'Linear momentum compaction {section.alpha_0}')
print()


# To declare a RingSection with time based program as input
length = 300  # m
alpha_0 = 1e-3
momentum = [[0, 1, 2],
            [26e9, 27e9, 28e9]]  # [s, eV/c]

section = RingSection(length, alpha_0, momentum)

print('-- Programs - Time based')
print(f'Section length {section.length} [m]')
print(f'Sync. data - Momentum {section.synchronous_data} [eV/c]')
print(f'Linear momentum compaction {section.alpha_0}')
print()


# To declare a RingSection using the input program function, time based
length = 300  # m
alpha_0 = 1e-3
momentum = machine_program([[0, 1, 2],
                            [26e9, 27e9, 28e9]],
                           interpolation='linear')  # [s, eV/c]

section = RingSection(length, alpha_0, momentum)

print('-- Programs - Time based with machine_program')
print(f'Section length {section.length} [m]')
print(f'Sync. data - Momentum {section.synchronous_data} [eV/c]')
print(f'Linear momentum compaction {section.alpha_0}')
print()

# To declare a RingSection with a programmed orbit bump
length = 300  # m
alpha_0 = 1e-3
bending_field = [1.] * 3  # T
bending_radius = 749  # m
orbit_bump = [0.001, 0.002, 0.003]  # m

section = RingSection(length, alpha_0,
                      bending_field=bending_field,
                      bending_radius=bending_radius,
                      orbit_bump=orbit_bump)

print('-- Orbit bump')
print(f'Section length {section.length_design} [m]')
print(f'Section length with orbit bump {section.length} [m]')
print(f'Orbit bump {section.orbit_bump} [m]')
print(f'Sync. data - Momentum {section.synchronous_data} [eV/c]')
print(f'Linear momentum compaction {section.alpha_0}')
print()


# To declare a RingSection passing non-linear momentum compaction factors
length = 300  # m
alpha_0 = 1e-3
alpha_1 = 1e-4
alpha_2 = 1e-5
alpha_5 = 1e-9
energy = machine_program([[0, 1, 2],
                          [26e9, 27e9, 28e9]])  # [s, eV]

section = RingSection(length, alpha_0, energy=energy,
                      alpha_1=alpha_1, alpha_2=alpha_2, alpha_5=alpha_5)

print('-- Non-linear momentum compaction factor')
print(f'Section length {section.length_design} [m]')
print(f'Sync. data - Momentum {section.synchronous_data} [eV/c]')
print(f'Linear momentum compaction {section.alpha_0}')
print(f'Nonlinear momentum compaction 1 {section.alpha_1}')
print(f'Nonlinear momentum compaction 2 {section.alpha_2}')
print(f'Nonlinear momentum compaction 3 {section.alpha_3}')
print(f'Nonlinear momentum compaction 5 {section.alpha_5}')
print(f'Orders defined {section.alpha_orders}')
print()
