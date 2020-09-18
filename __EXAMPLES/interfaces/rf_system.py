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
from blond_common.interfaces.machine_parameters.rf_system import \
    RFSystem


# To declare an RFSystem with simple input, using rf harmonic
voltage = 1e3  # V
phase = np.pi  # rad
harmonic = 2

rf_system = RFSystem(voltage, phase, harmonic)

print('-- Simple input, rf harmonic')
print(f'RF voltage {rf_system.voltage} [V]')
print(f'RF phase {rf_system.phase} [rad]')
print(f'RF harmonic {rf_system.harmonic}')
print(f'RF frequency {rf_system.frequency} [Hz]')
print()


# To declare an RFSystem with simple input, using fixed rf frequency
voltage = 1e3  # V
phase = np.pi  # rad
frequency = 1e3  # Hz

rf_system = RFSystem(voltage, phase, frequency=frequency)

print('-- Simple input, fixed rf frequency')
print(f'RF voltage {rf_system.voltage} [V]')
print(f'RF phase {rf_system.phase} [rad]')
print(f'RF harmonic {rf_system.harmonic}')
print(f'RF frequency {rf_system.frequency} [Hz]')
print()


# To declare an RFSystem with turn-by-turn program as input
voltage = [1e3, 1.1e3, 1.2e3]  # V
phase = [np.pi, np.pi - 0.1, np.pi - 0.2]  # rad
harmonic = 2

rf_system = RFSystem(voltage, phase, harmonic)

print('-- Turn-by-turn input, rf harmonic')
print(f'RF voltage {rf_system.voltage} [V]')
print(f'RF phase {rf_system.phase} [rad]')
print(f'RF harmonic {rf_system.harmonic}')
print(f'RF frequency {rf_system.frequency} [Hz]')
print()


# To declare an RFSystem with time-based program as input
voltage = [[0, 1, 2],
           [1e3, 1.1e3, 1.2e3]]  # [s, V]
phase = np.pi  # rad
frequency = [[0, 1, 2],
             [1e3, 1.1e3, 1.2e3]]  # [s, Hz]

rf_system = RFSystem(voltage, phase, frequency=frequency)

print('-- Time-based input, rf frequency')
print(f'RF voltage {rf_system.voltage} [s, V]')
print(f'RF phase {rf_system.phase} [rad]')
print(f'RF harmonic {rf_system.harmonic}')
print(f'RF frequency {rf_system.frequency} [s, Hz]')
print()


# To declare many RF systems with time based programs, and combining
# them into one representing the total voltage seen by the beam

harmonic_1 = [[0, 1, 2], [2, 2, 2]]  # [s, unitless]
voltage_1 = [[0, 1.5, 2], [1e3, 1.1e3, 1.2e3]]  # [s, V]
phase_1 = np.pi  # rad

harmonic_2 = harmonic_1  # [s, unitless]
voltage_2 = [[0, 1, 2], [1e3, 1.1e3, 1.2e3]]  # [s, V]
phase_2 = [[0, 0.5, 2], [np.pi, np.pi - 0.1, np.pi - 0.2]]  # [s, rad]

rf_system_1 = RFSystem(voltage_1, phase_1, harmonic_1)
rf_system_2 = RFSystem(voltage_2, phase_2, harmonic_2)

global_rf_system = RFSystem.combine_systems([rf_system_1, rf_system_2])[0]

print('-- Combined rf system - two RF')
print(f'RF voltage {global_rf_system.voltage} [s, V]')
print(f'RF phase {global_rf_system.phase} [rad]')
print(f'RF harmonic {global_rf_system.harmonic}')
print(f'RF frequency {global_rf_system.frequency} [s, Hz]')


# To declare many RF systems with time based programs, and combining
# them into one representing the total voltage seen by the beam

harmonic_1 = [[0, 1, 2], [2, 2, 2]]  # [s, unitless]
voltage_1 = [[0, 1.5, 2], [1e3, 1.1e3, 1.2e3]]  # [s, V]
phase_1 = np.pi  # rad

harmonic_2 = [[0, 1, 2], [2, 2, 2]]  # [s, unitless]
voltage_2 = [[0, 1, 2], [1e3, 1.1e3, 1.2e3]]  # [s, V]
phase_2 = [[0, 0.5, 2], [np.pi, np.pi - 0.1, np.pi - 0.2]]  # [s, rad]

harmonic_3 = [[0, 1, 2], [3, 3, 3]]  # [s, unitless]
voltage_3 = [[0, 1, 2], [0.5e3, 1.0e3, 1.5e3]]  # [s, V]
phase_3 = [[0, 1.5, 2], [0., 0.1, 0.2]]  # [s, rad]

freq_4 = [[0, 1, 2], [1e3, 1.1e3, 1.2e3]]  # [s, Hz]
voltage_4 = [[0, 1, 2], [0.5e3, 1.0e3, 1.5e3]]  # [s, V]
phase_4 = [[0, 1.5, 2], [0., 0.1, 0.2]]  # [s, rad]

rf_system_1 = RFSystem(voltage_1, phase_1, harmonic_1)
rf_system_2 = RFSystem(voltage_2, phase_2, harmonic_2)
rf_system_3 = RFSystem(voltage_3, phase_3, harmonic_3)
rf_system_4 = RFSystem(voltage_4, phase_4, frequency=freq_4)

global_rf_system_1, global_rf_system_2, global_rf_system_3 = \
    RFSystem.combine_systems([rf_system_1, rf_system_2,
                              rf_system_3, rf_system_4])

print('-- Combined rf system - many RF - 1')
print(f'RF voltage {global_rf_system_1.voltage} [s, V]')
print(f'RF phase {global_rf_system_1.phase} [rad]')
print(f'RF harmonic {global_rf_system_1.harmonic}')
print(f'RF frequency {global_rf_system_1.frequency} [s, Hz]')
print()
print('-- Combined rf system - many RF - 2')
print(f'RF voltage {global_rf_system_2.voltage} [s, V]')
print(f'RF phase {global_rf_system_2.phase} [rad]')
print(f'RF harmonic {global_rf_system_2.harmonic}')
print(f'RF frequency {global_rf_system_2.frequency} [s, Hz]')
print()
print('-- Combined rf system - many RF - 3')
print(f'RF voltage {global_rf_system_3.voltage} [s, V]')
print(f'RF phase {global_rf_system_3.phase} [rad]')
print(f'RF harmonic {global_rf_system_3.harmonic}')
print(f'RF frequency {global_rf_system_3.frequency} [s, Hz]')
print()
