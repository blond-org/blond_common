# coding: utf8
# Copyright 2014-2020 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Module handling all parameters related to the RFSystem object.**
    :Authors: **Simon Albright**, **Alexandre Lasheen**
'''


# General imports
import numpy as np
import warnings

# BLonD_Common imports
from ...datatypes import rf_programs, blond_function
from ...devtools import assertions as assrt
from ...devtools import exceptions as excpt


class RFSystem:
    r""" Class containing the general properties of an RF system, regardless
    of its position in the Ring.

    The user has to input the rf harmonic or fixed rf frequency with at least
    but no more than one of the following input: harmonic, frequency.

    Parameters
    ----------
    voltage : float (opt: list or np.ndarray)
        The input rf voltage in [V]
        Can be input as a single float value or as a program
        (1D is turn-by-turn, 2D is time based).
    phase : float (opt: list or np.ndarray)
        The input rf phase in [rad].
        Can be input as a single float value or as a program
        (1D is turn-by-turn, 2D is time based)
    harmonic : float (opt: list or np.ndarray)
        The input rf harmonic
        Either the harmonic or the frequency can be passed as input
        Can be input as a single float value or as a program
        (1D is turn-by-turn, 2D is time based)
    frequency : float (opt: list or np.ndarray)
        The input rf frequency in [Hz] (can be non integer harmonic of the
        revolution frequency)
        Either the harmonic or the frequency can be passed as input
        Can be input as a single float value or as a program
        (1D is turn-by-turn, 2D is time based).

    Attributes
    ----------
    voltage : datatype.rf_programs.voltage_program
        The input rf voltage in [V]
    phase : datatype.rf_programs.voltage_program
        The input rf phase in [rad]
    harmonic : datatype.blond_functions.machine_program or None
        The input rf harmonic
    frequency : datatype.blond_functions.machine_program or None
        The input rf frequency in [Hz]

    Examples
    --------
    >>> # General imports
    >>> import numpy as np

    >>> # To declare an RF system with very simple parameters
    >>> from blond_common.interfaces.machine_parameters.rf_system import \
    >>>     RFSystem
    >>>
    >>> voltage = 1e3
    >>> phase = np.pi
    >>> harmonic = 2
    >>>
    >>> rf_system = RFSystem(voltage, phase, harmonic)

    >>> # To declare two RF systems with time based programs, and combining
    >>> # them into one representing the total voltage seen by the beam
    >>>
    >>> harmonic = 2
    >>>
    >>> voltage_1 = [[0, 1, 2], [1e3, 1.1e3, 1.2e3]]
    >>> phase_1 = np.pi
    >>>
    >>> voltage_2 = [[0, 1, 2], [1e3, 1.1e3, 1.2e3]]
    >>> phase_2 = [[0, 1, 2], [np.pi, np.pi-0.1, np.pi-0.2]]
    >>>
    >>> rf_system_1 = RFSystem(voltage_1, phase_1, harmonic)
    >>> rf_system_2 = RFSystem(voltage_2, phase_2, harmonic)
    >>>
    >>> global_rf_system = RFSystem.combine_systems([rf_system_1, rf_system_2])
    """

    def __init__(self, voltage, phase, harmonic=None, frequency=None):

        # Checking that at least one rf frequency data input is passed
        freqDataTypes = ('harmonic', 'frequency')
        freqDataInput = (harmonic, frequency)
        assrt.single_not_none(*freqDataInput,
                              msg='Exactly one of ' + str(freqDataTypes) +
                              ' must be declared',
                              exception=excpt.InputError)

        # Passing harmonic and frequency as datatypes
        # Using blond_function.machine_program for now in absence
        # of dedicated programs
        if not isinstance(harmonic, blond_function.machine_program):
            harmonic = rf_programs._RF_function(
                harmonic, harmonics=[None])
        self.harmonic = harmonic

        if not isinstance(frequency, blond_function.machine_program):
            frequency = rf_programs._RF_function(
                frequency, harmonics=[None])
        self.frequency = frequency

        # Ref data for _check_and_set
        if frequency is None:
            ref_data_freq = 'harmonic'
        else:
            ref_data_freq = 'frequency'

        # Getting the harmonics to be passed to the voltage/phase datatypes
        if self.harmonic.timebase == 'by_time':
            unique_harmonics = np.unique(self.harmonic[:, 1, :])
        else:
            unique_harmonics = np.unique(self.harmonic)
        if (len(unique_harmonics) > 1) or (unique_harmonics[0] is None):
            unique_harmonics = None
        else:
            unique_harmonics = int(unique_harmonics)

        # Setting the voltage program as a datatypes.voltage_program
        if not isinstance(voltage, rf_programs.voltage_program):
            voltage = rf_programs.voltage_program(
                voltage, harmonics=unique_harmonics)
        self._check_and_set_rf_prog(voltage, 'voltage', ref_data_freq)

        # Setting the phase program as a datatypes.phase_program
        if not isinstance(phase, rf_programs.phase_program):
            phase = rf_programs.phase_program(
                phase, harmonics=unique_harmonics)
        self._check_and_set_rf_prog(phase, 'phase', ref_data_freq)
        self._check_and_set_rf_prog(phase, 'phase', 'voltage')

    def sample(self, use_time=None, use_turns=None):

        voltage = self.voltage.reshape([self.harmonic], use_time, use_turns)
        phase = self.phase.reshape([self.harmonic], use_time, use_turns)

        return voltage, phase, self.harmonic

    @classmethod
    def combine_systems(cls, RFSystem_list):

        # Initializing lists where the combined data from the systems
        # are stored
        combined_system_list = []
        unique_constant_harmonics = []
        voltage_per_harmonic = []
        phase_per_harmonic = []

        for single_system in RFSystem_list:

            # Extracing the harmonic from the single_system (datatype)
            if single_system.harmonic.timebase == 'by_time':
                hamonic_values = single_system.harmonic[:, 1, :]
            else:
                hamonic_values = single_system.harmonic

            # Finding the unique values of harmonics in the system
            single_system_unique_harmonics = np.unique(hamonic_values)

            # Not combining the system with others if harmonic is a varying
            # program or if the system is defined using fixed frequency
            if (len(single_system_unique_harmonics) > 1) or \
                    (single_system_unique_harmonics[0] is None):
                combined_system_list.append(single_system)
            else:

                single_system_unique_harmonics.timebase = 'single'
                if single_system_unique_harmonics not in \
                        unique_constant_harmonics:

                    unique_constant_harmonics.append(
                        single_system_unique_harmonics)

                    voltage_per_harmonic.append(
                        [single_system.voltage[i].view(np.ndarray)
                         for i in range(single_system.voltage.shape[0])])

                    phase_per_harmonic.append(
                        [single_system.phase[i].view(np.ndarray)
                         for i in range(single_system.phase.shape[0])])
                else:

                    idx_harmonic = np.where(
                        single_system_unique_harmonics ==
                        unique_constant_harmonics)[0][0]

                    voltage_per_harmonic[idx_harmonic] += [
                        single_system.voltage[i].view(np.ndarray)
                        for i in range(single_system.voltage.shape[0])]

                    phase_per_harmonic[idx_harmonic] += [
                        single_system.phase[i].view(np.ndarray)
                        for i in range(single_system.phase.shape[0])]

        # Regenerating RFSystems with n-dimensional voltage and phase programs
        for idx_combined in range(len(unique_constant_harmonics)):
            combined_system_list.append(cls(
                rf_programs.voltage_program(
                    *voltage_per_harmonic[idx_combined],
                    harmonics=[int(unique_constant_harmonics[idx_combined])] *
                    len(voltage_per_harmonic[idx_combined])),
                rf_programs.phase_program(
                    *phase_per_harmonic[idx_combined],
                    harmonics=[int(unique_constant_harmonics[idx_combined])] *
                    len(phase_per_harmonic[idx_combined])),
                unique_constant_harmonics[idx_combined]))

        return combined_system_list

    def _check_and_set_rf_prog(self, rf_prog, prog_type, reference_prog):
        '''
        Internal function to check that the input rf program length is
        coherent with the reference program. If the reference program
        is turn based, the rf program should have the same length.
        If the reference program is time based while the rf program is turn
        based, raises a warning.
        '''

        setattr(self, prog_type, rf_prog)
        ref_attr = getattr(self, reference_prog)

        if (ref_attr.timebase == 'by_turn') and \
                (rf_prog.timebase == 'by_turn'):

            if (rf_prog.shape[-1] > 1) and \
                    (ref_attr.shape[-1]
                     > rf_prog.shape[-1]):

                raise excpt.InputError(
                    'The input ' + prog_type +
                    ' was passed as a turn based program but with ' +
                    'different length than the ' + reference_prog +
                    ' program. Turn based programs should have the same ' +
                    'length.')

        elif (ref_attr.timebase == 'by_time') and \
                (rf_prog.timebase == 'by_turn'):

            warn_message = 'The ' + reference_prog + ' was defined time based while the ' + \
                'input ' + prog_type + ' was defined turn base, this may' + \
                'lead to errors in the RFStation object after interpolation.'
            warnings.warn(warn_message)

        elif (ref_attr.timebase == 'by_turn') and \
                (rf_prog.timebase == 'by_time'):

            warn_message = 'The ' + reference_prog + ' was defined turn based while the ' + \
                'input ' + prog_type + ' was defined time base, this may' + \
                'lead to errors in the RFStation object after interpolation.'
            warnings.warn(warn_message)
