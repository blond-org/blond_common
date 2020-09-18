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


# To declare two RF systems with time based programs, and combining
# them into one representing the total voltage seen by the beam

harmonic = [[0, 1, 2], [2, 2, 2]]

voltage_1 = [[0, 1.5, 2], [1e3, 1.1e3, 1.2e3]]
phase_1 = np.pi

voltage_2 = [[0, 1, 2], [1e3, 1.1e3, 1.2e3]]
phase_2 = [[0, 0.5, 2], [np.pi, np.pi - 0.1, np.pi - 0.2]]

rf_system_1 = RFSystem(voltage_1, phase_1, harmonic)
rf_system_2 = RFSystem(voltage_2, phase_2, [[0, 1, 2], [3, 3, 3]])

global_rf_system = RFSystem.combine_systems([rf_system_1, rf_system_2])[0]

print('-- Combined rf system')
print(f'RF voltage {global_rf_system.voltage} [s, V]')
print(f'RF phase {global_rf_system.phase} [rad]')
print(f'RF harmonic {global_rf_system.harmonic}')
print(f'RF frequency {global_rf_system.frequency} [s, Hz]')
print()
