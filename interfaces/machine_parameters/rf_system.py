# coding: utf8
# Copyright 2014-2020 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Module gathering all general input parameters used for the simulation.**
    :Authors: **Simon Albright**, **Alexandre Lasheen**
'''

# BLonD_Common imports
from ...datatypes import rf_programs


class RFSystem:
    r""" Class containing the general properties of an RF system, regardless
    of its position in the Ring.

    The user has to input the rf harmonic or fixed rf frequency with at least
    but no more than one of the following input: harmonic, frequency.

    Parameters
    ----------
    voltage : float (opt: list or np.ndarray)
        Length [m] of accelerator section on the reference orbit (see
        orbit_length option).
    phase : float (opt: list or np.ndarray)
        Length [m] of accelerator section on the reference orbit (see
        orbit_length option).
    harmonic : float (opt: list or np.ndarray)
        Length [m] of accelerator section on the reference orbit (see
        orbit_length option).
    frequency : float (opt: list or np.ndarray)
        Length [m] of accelerator section on the reference orbit (see
        orbit_length option).

    Attributes
    ----------
    voltage : datatype.rf_programs.voltage_program
        RF voltage program [V]
    phase : datatype.rf_programs.voltage_program
        RF voltage program [V]
    voltage : datatype.rf_programs.voltage_program
        RF voltage program [V]

    Examples
    --------
    >>> # General improts
    >>> import numpy as np
    >>>
    >>> # To declare an RF system with very simple parameters
    >>> from blond_common.interfaces.machine_parameters.rf_system import \
    >>>     RFSystem
    >>>
    >>> voltage = 1e3
    >>> phase = np.pi
    >>> harmonic = 2
    >>>
    >>> rf_system = RFSystem(voltage, phase, harmonic)
    """

    def __init__(self, voltage, phase, harmonic=None, frequency=None):

        if not isinstance(voltage, rf_programs.voltage_program):
            voltage = rf_programs.voltage_program(
                voltage, harmonics=[harmonic])
        self.voltage = voltage

        if not isinstance(phase, rf_programs.phase_program):
            phase = rf_programs.phase_program(phase, harmonics=[harmonic])
        self.phase = phase

        self.harmonic = harmonic

    def sample(self, use_time=None, use_turns=None):

        voltage = self.voltage.reshape([self.harmonic], use_time, use_turns)
        phase = self.phase.reshape([self.harmonic], use_time, use_turns)

        return voltage, phase, self.harmonic

    @classmethod
    def combine_systems(cls, RFSystem_list):

        pass
