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

# BLonD Common import
from blond_common.interfaces.machine_parameters.ring import \
    Ring, RingSection, machine_program
from blond_common.interfaces.beam.beam import Proton, Electron, Particle


# To declare a Ring with simple input
length = 628  # m
alpha_0 = 1e-3
momentum = 26e9  # eV/c
particle = Proton()

ring = Ring(particle, [RingSection(length, alpha_0, momentum)])
# ring = Ring.direct_input(particle, length, alpha_0, momentum)

print('-- Simple input')
print(f'Ring circumference {ring.circumference_design} [m]')
print(f'Sync. data - Momentum {ring.momentum} [eV/c]')
print(f'Sync. data - Beta {ring.beta}')
print(f'Sync. data - Gamma {ring.gamma}')
print(f'Linear momentum compaction {ring.alpha_0}')
print()


# To declare a Ring with other synchronous data (all possible definitions
# from the RingSection object, example with total energy)
length = 628  # m
alpha_0 = 1e-3
tot_energy = 26e9  # eV/c
particle = 'proton'

section = RingSection(length, alpha_0, energy=tot_energy)
ring = Ring(particle, section)

# ring = Ring.direct_input(particle, length, alpha_0, energy=tot_energy)

print('-- Other sync. data - Total energy')
print(f'Ring circumference {ring.circumference_design} [m]')
print(f'Sync. data - Momentum {ring.momentum} [eV/c]')
print(f'Sync. data - Total energy {ring.energy} [eV]')
print(f'Sync. data - Beta {ring.beta}')
print(f'Sync. data - Gamma {ring.gamma}')
print(f'Linear momentum compaction {ring.alpha_0}')
print()

# To declare a Ring with other synchronous data (all possible definitions
# from the RingSection object, example with bending field)
length = 628  # m
alpha_0 = 1e-3
bending_field = 1  # T
bending_radius = 70  # m
particle = 'proton'

section = RingSection(length, alpha_0, bending_field=bending_field,
                      bending_radius=bending_radius)
ring = Ring(particle, section)

# ring = Ring.direct_input(particle, length, alpha_0,
#                          bending_field=bending_field,
#                          bending_radius=bending_radius)

print('-- Other sync. data - Bending field')
print(f'Ring circumference {ring.circumference_design} [m]')
print(f'Sync. data - Momentum {ring.momentum} [eV/c]')
print(f'Sync. data - Total energy {ring.energy} [eV]')
print(f'Sync. data - Beta {ring.beta}')
print(f'Sync. data - Gamma {ring.gamma}')
print(f'Linear momentum compaction {ring.alpha_0}')
print()


# To declare a Ring with other particle type (electrons)
length = 80e3  # m
alpha_0 = 1e-3
momentum = 120e9  # eV/c
particle = Electron()  # can also be 'electron'

section = RingSection(length, alpha_0, momentum)
ring = Ring(particle, section)

# ring = Ring.direct_input(particle, length, alpha_0, momentum)

print('-- Other particle type - Electrons')
print(f'Particle charge {ring.Particle.charge} [e]')
print(f'Particle mass energy {ring.Particle.mass} [eV]')
print(f'Ring circumference {ring.circumference_design} [m]')
print(f'Sync. data - Momentum {ring.momentum} [eV/c]')
print(f'Sync. data - Total energy {ring.energy} [eV]')
print(f'Sync. data - Beta {ring.beta}')
print(f'Sync. data - Gamma {ring.gamma}')
print(f'Linear momentum compaction {ring.alpha_0}')
print()


# To declare a Ring with other particle type (ions)
length = 6.9e3
alpha_0 = 1e-3
user_mass = 208 * Proton().mass
user_charge = 82
momentum = user_charge * 450e9  # eV/c
particle = Particle(user_mass, user_charge)

section = RingSection(length, alpha_0, momentum)
ring = Ring(particle, section)

# ring = Ring.direct_input(particle, length, alpha_0, momentum)

print('-- Other particle type - Ions')
print(f'Particle charge {ring.Particle.charge} [e]')
print(f'Particle mass energy {ring.Particle.mass} [eV]')
print(f'Ring circumference {ring.circumference_design} [m]')
print(f'Sync. data - Momentum {ring.momentum} [eV/c]')
print(f'Sync. data - Total energy {ring.energy} [eV]')
print(f'Sync. data - Beta {ring.beta}')
print(f'Sync. data - Gamma {ring.gamma}')
print(f'Linear momentum compaction {ring.alpha_0}')
print()


# To declare a Ring with non-linear momentum compaction factor
length = 628  # m
alpha_0 = 1e-3
alpha_1 = 1e-6
momentum = 26e9  # eV/c
particle = Proton()

section = RingSection(length, alpha_0, momentum, alpha_1=alpha_1)
ring = Ring(particle, section)

# ring = Ring.direct_input(particle, length, alpha_0, momentum,
#                          alpha_1=alpha_1)

print('-- Non-linear momentum compaction factor')
print(f'Ring circumference {ring.circumference} [m]')
print(f'Sync. data - Momentum {ring.momentum} [eV/c]')
print(f'Linear momentum compaction {ring.alpha_0}')
print(f'Non-linear momentum compaction 1 {ring.alpha_1}')
print()


# To declare a Ring with a ramp (turn based program)
length = 628  # m
alpha_0 = 1e-3
momentum = [26e9, 27e9, 28e9]  # eV/c
particle = Proton()

section = RingSection(length, alpha_0, momentum)
ring = Ring(particle, section)

# ring = Ring.direct_input(particle, length, alpha_0, momentum)

print('-- Programs - Turn based')
print(f'Ring circumference {ring.circumference_design} [m]')
print(f'Cycle time {ring.cycle_time} [s]')
print(f'Sync. data - Momentum {ring.momentum} [eV/c]')
print(f'Linear momentum compaction {ring.alpha_0}')
print()


# To declare a Ring with a ramp (turn based program)
length = 628  # m
alpha_0 = 1e-3
momentum = machine_program(26e9, n_turns=10)  # eV/c
particle = Proton()

section = RingSection(length, alpha_0, momentum)
ring = Ring(particle, section)

# ring = Ring.direct_input(particle, length, alpha_0, momentum)

print('-- Programs - Turn based')
print(f'Ring circumference {ring.circumference_design} [m]')
print(f'Cycle time {ring.cycle_time} [s]')
print(f'Sync. data - Momentum {ring.momentum} [eV/c]')
print(f'Linear momentum compaction {ring.alpha_0}')
print()


# To declare a Ring with a ramp (time based program)
length = 628  # m
alpha_0 = 1e-3
momentum = [[0, 0.1, 0.2], [26e9, 27e9, 28e9]]  # [s, eV/c]
particle = Proton()

section = RingSection(length, alpha_0, momentum)
ring = Ring(particle, section)

# ring = Ring.direct_input(particle, length, alpha_0, momentum)

print('-- Programs - Time based')
print(f'Ring circumference {ring.circumference_design} [m]')
print(f'Cycle time {ring.cycle_time} [s]')
print(f'Sync. data - Momentum {ring.momentum} [eV/c]')
print(f'Energy gain {ring.delta_E} [eV/c]')
print(f'Linear momentum compaction {ring.alpha_0}')
print()


# To declare a Ring with time based programs (multiple, including orbit bump)
length = 628
alpha_0 = [[0, 0.1, 0.2], [1e-3, 1.1e-3, 1e-3]]
alpha_1 = [[0, 0.1, 0.2], [1e-6, 0.9e-6, 1e-6]]
momentum = [[0, 0.1, 0.2], [26e9, 26e9, 26e9]]
orbit_bump = [[0, 0.1, 0.2], [0., 1e-3, 0.]]
particle = Proton()

section = RingSection(length, alpha_0, momentum, alpha_1=alpha_1,
                      orbit_bump=orbit_bump)
ring = Ring(particle, section, eta_orders=1, interpolation='derivative',
            t_start=0.01)

ring = Ring.direct_input(particle, length, alpha_0, momentum,
                         alpha_1=alpha_1,
                         eta_orders=1, interpolation='derivative',
                         t_start=0.01)

print('-- Programs - Multiple time based programs and orbit bump')
print(f'Ring circumference {ring.circumference_design} [m]')
print(f'Ring circumference with bump {ring.circumference} [m]')
print(f'Revolution frequency {ring.f_rev_design} [Hz]')
print(f'Revolution frequency with bump {ring.f_rev} [Hz]')
print(f'Orbit bump {ring.orbit_bump} [m]')
print(f'Cycle time {ring.cycle_time} [s]')
print(f'Sync. data - Momentum {ring.momentum} [eV/c]')
print(f'Energy gain {ring.delta_E[0:9]} [eV/c]')
print(f'Linear momentum compaction {ring.alpha_0}')
print(f'Non-linear momentum compaction 1 {ring.alpha_1}')
print(f'Linear slippage factor {ring.eta_0}')
print(f'Non-linear slippage factor 1 {ring.eta_1}')
print()


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

print('-- Multiple sections')
print(f'Ring circumference {ring.circumference_design} [m]')
print(f'Section lengths {ring.section_length_design} [m]')
print(f'Sync. data - Momentum {ring.momentum} [eV/c]')
print(f'Linear momentum compaction {ring.alpha_0}')
print()


# To declare a Ring with multiple sections and different momentum program
length = 628 / 2
alpha_0 = 1e-3
momentum_1 = [26e9, 27e9, 28e9]
momentum_2 = [26.5e9, 27.5e9, 28.5e9]
particle = Proton()

section_1 = RingSection(length, alpha_0, momentum_1)
section_2 = RingSection(length, alpha_0, momentum_2)
ring = Ring(particle, [section_1, section_2])

print('-- Multiple sections')
print(f'Ring circumference {ring.circumference_design} [m]')
print(f'Section lengths {ring.section_length_design} [m]')
print(f'Sync. data - Momentum {ring.momentum} [eV/c]')
print(f'Linear momentum compaction {ring.alpha_0}')
print()
