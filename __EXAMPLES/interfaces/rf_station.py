# coding: utf8
# Copyright 2014-2020 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Examples of usage of the RFSystem object.**
    :Authors: **Alexandre Lasheen**
'''

# General imports
import numpy as np

# BLonD Common import
# Add blond_common to PYTHONPATH if not done
from blond_common.interfaces.machine_parameters.ring import \
    Ring, RingSection
from blond_common.interfaces.machine_parameters.rf_station import \
    RFSystem, RFStation
from blond_common.datatypes import blond_function, rf_programs

# Generate a base Ring object with 1xRingSection
length = 6911  # m
alpha_0 = 1 / 18**2.
momentum = [[0, 1, 2], [26e9, 27e9, 28e9]]  # [s, eV/c]
particle = 'proton'

ring = Ring(particle, [RingSection(length, alpha_0, momentum)])


# To declare an RFSystem with simple input, using rf harmonic
voltage = 4.5e6  # V
phase = np.array([[0, 1, 2], [0, np.pi / 2, np.pi]])  # rad
phase_2 = np.array([[0, 1, 2], [0, np.pi / 2, np.pi]])  # rad
phase_2[1, :] += np.pi
harmonic = [[0, 1, 2], [4620, 4620, 4620]]
f_rf = 200e6  # Hz

rf_station = RFStation(ring,
                       [RFSystem(voltage, phase, harmonic),
                        RFSystem(voltage, phase_2, harmonic),
                        RFSystem(voltage, phase, frequency=f_rf)],
                       combine_systems=True)

# rf_station = RFStation.direct_input()

print('-- Complex input 1')
print(f'RF voltage {rf_station.voltage} [V]')
print(f'RF phase {rf_station.phi_rf} [rad]')
print(f'RF harmonic {rf_station.harmonic}')
print(f'RF frequency {rf_station.f_rf} [Hz]')
print()


# To declare many RF systems with time based programs, and combining
# them into one representing the total voltage seen by the beam

harmonic_1 = [[0, 1, 2], [2, 2, 2]]  # [s, unitless]
voltage_1 = [[0, 1.5, 2], [1e3, 1.1e3, 1.2e3]]  # [s, V]
phase_1 = np.pi  # rad

harmonic_2 = [[0, 1, 2], [2, 2, 2]]  # [s, unitless]
voltage_2 = [[0, 1, 2], [1e3, 1.1e3, 1.2e3]]  # [s, V]
phase_2 = [[0, 0.5, 2], [np.pi, np.pi - 0.1, np.pi - 0.2]]  # [s, rad]

harmonic_3 = [[0, 1, 2], [2, 3, 3]]  # [s, unitless]
voltage_3 = [[0, 1, 2], [0.5e3, 1.0e3, 1.5e3]]  # [s, V]
phase_3 = [[0, 1.5, 2], [0., 0.1, 0.2]]  # [s, rad]

freq_4 = [[0, 1, 2], [100e3, 100.1e3, 100.2e3]]  # [s, Hz]
voltage_4 = [[0, 1, 2], [0.5e3, 1.0e3, 1.5e3]]  # [s, V]
phase_4 = [[0, 1.5, 2], [0., 0.1, 0.2]]  # [s, rad]

rf_system_1 = RFSystem(voltage_1, phase_1, harmonic_1)
rf_system_2 = RFSystem(voltage_2, phase_2, harmonic_2)
rf_system_3 = RFSystem(voltage_3, phase_3, harmonic_3)
rf_system_4 = RFSystem(voltage_4, phase_4, frequency=freq_4)

rf_station = RFStation(ring,
                       [rf_system_1,
                        rf_system_2,
                        rf_system_3,
                        rf_system_4],
                       combine_systems=True)

# rf_station = RFStation.direct_input()

print('-- Complex input 2')
print(f'RF voltage {rf_station.voltage} [V]')
print(f'RF phase {rf_station.phi_rf} [rad]')
print(f'RF harmonic {rf_station.harmonic}')
print(f'RF frequency {rf_station.f_rf} [Hz]')
print()
