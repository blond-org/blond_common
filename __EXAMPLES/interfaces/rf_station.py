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
from blond_common.interfaces.machine_parameters.rf_station import \
    RFSystem, RFStation


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
