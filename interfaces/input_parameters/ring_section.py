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

# General imports
import numpy as np
import warnings

# BLonD_Common imports
from ...datatypes import ring_programs
from ...datatypes.blond_function import machine_program
from ...devtools import exceptions as excpt
from ...devtools import assertions as assrt


class RingSection:
    r""" Class containing the general properties of a section of the
    accelerator that are independent of the RF system or the beam.

    The user has to input the synchronous data with at least but no more
    than one of the following: momentum, kin_energy, energy, bending_field.

    If bending_field is passed, bending_radius should be passed as well.

    Parameters
    ----------
    length : float
        Length [m] of accelerator section on the reference orbit (see
        orbit_length option).
    alpha_0 : float (opt: list or np.ndarray)
        Momentum compaction factor of zeroth order :math:`\alpha_{0}`;
        can be input as single float or as a program (1D array is a
        turn-by-turn program and 2D array is a time dependent program).
        If a turn-by-turn program is passed, should be of the same size
        as the synchronous data.
    momentum : float (opt: list or np.ndarray)
        Design particle momentum [eV]
        on the design orbit :math:`p_{s}`;
        can be input as single float or as a program (1D array is a
        turn-by-turn program and 2D array is a time dependent program).
    kin_energy : float (opt: list or np.ndarray)
        Design particle kinetic energy [eV]
        on the design orbit :math:`E_{k,s}`;
        can be input as single float or as a program (1D array is a
        turn-by-turn program and 2D array is a time dependent program).
    energy : float (opt: list or np.ndarray)
        Design particle total energy [eV]
        on the design orbit :math:`E_{s}`;
        can be input as single float or as a program (1D array is a
        turn-by-turn program and 2D array is a time dependent program).
    bending_field : float (opt: list or np.ndarray)
        Design bending field [T] on the design orbit :math:`B_{s}`,
        this is used to defined the design particle momentum;
        can be input as single float or as a program (1D array is a
        turn-by-turn program and 2D array is a time dependent program).
    bending_radius : float
        Optional: Radius [m] of the bending magnets,
        required if 'bending field' is set for the synchronous_data_type
    orbit_bump : float (opt: list or np.ndarray)
        Length of orbit bump to add as increment to the design length;
        can be input as single float or as a program (1D array is a
        turn-by-turn program and 2D array is a time dependent program).
        If a turn-by-turn program is passed, should be of the same size
        as the synchronous data.
    alpha_n : float (opt: list or np.ndarray)
        Optional : Higher order momentum compaction can also be passed through
        keyword arguments (orders not limited, e.g. alpha_3 is recognized);
        can be input as single float or as a program (1D array is a
        turn-by-turn program and 2D array is a time dependent program).
        If a turn-by-turn program is passed, should be of the same size
        as the synchronous data.

    Attributes
    ----------
    length_design : float
        Length of the section on the reference orbit [m]
    length : datatype.machine_program.orbit_length
        Length of the beam trajectory, including possible
        orbit bump programs [m]
    synchronous_data : datatype.machine_program._ring_program
        The user input synchronous data, with no conversion applied.
        The datatype depends on the user input and can be
        momentum_program, kinetic_energy_program, total_energy_program,
        bending_field_program
    bending_radius : float (or None)
        Bending radius in dipole magnets, :math:`\rho` [m]
    alpha_0 : datatype.machine_program.momentum_compaction
        Momentum compaction factor of zeroth order
    alpha_n : datatype.machine_program.momentum_compaction (or undefined)
        Momentum compaction factor of higher orders
    alpha_orders : int
        Orders of momentum compaction defined by the user

    Examples
    --------
    >>> # To declare a section of a synchrotron with very simple
    >>> # parameters
    >>> from blond_common.interfaces.input_parameters.ring_section import \
    >>>     Section
    >>>
    >>> length = 300
    >>> alpha_0 = 1e-3
    >>> momentum = 26e9
    >>>
    >>> section = RingSection(length, alpha_0, momentum)

    >>> # To declare a section of a synchrotron with very complex
    >>> # parameters and programs
    >>> from blond_common.interfaces.input_parameters.ring_section import \
    >>>     Section
    >>>
    >>> length = 300
    >>> alpha_0 = [1e-3, 0.9e-3, 1e-3]
    >>> alpha_1 = 1e-4
    >>> alpha_2 = 1e-5
    >>> alpha_5 = 1e-9
    >>> orbit_bump = [0., 1e-3, 0.]
    >>> energy = machine_program([[0, 1, 2],
    >>>                           [26e9, 27e9, 28e9]])
    >>>
    >>> section = RingSection(length, alpha_0, energy=energy,
    >>>                       alpha_1=alpha_1, alpha_2=alpha_2,
    >>>                       alpha_5=alpha_5, orbit_bump=orbit_bump)
    """

    def __init__(self, length, alpha_0,
                 momentum=None, kin_energy=None, energy=None,
                 bending_field=None, bending_radius=None, orbit_bump=None,
                 alpha_1=None, alpha_2=None, **kwargs):

        # Setting section length
        self.length_design = float(length)

        # Checking that at least one synchronous data input is passed
        syncDataTypes = ('momentum', 'kin_energy', 'energy', 'B_field')
        syncDataInput = (momentum, kin_energy, energy, bending_field)
        assrt.single_not_none(*syncDataInput,
                              msg='Exactly one of ' + str(syncDataTypes) +
                              ' must be declared',
                              exception=excpt.InputError)

        # Checking that the bending_radius is passed with the bending_field
        # and setting the bending_radius if defined
        if bending_field is not None and bending_radius is None:
            raise excpt.InputError("If bending_field is used, " +
                                   "bending_radius must be defined.")
        else:
            self.bending_radius = bending_radius

        # Taking the first synchronous_data input not declared as None
        # The assertion above ensures that only one is declared
        for func_type, synchronous_data in zip(syncDataTypes, syncDataInput):
            if synchronous_data is not None:
                break

        # Reshaping the input synchronous data to the adequate format and
        # get back the momentum program
        if not isinstance(synchronous_data,
                          ring_programs._synchronous_data_program):
            synchronous_data \
                = ring_programs._synchronous_data_program._conversions[
                    func_type](synchronous_data)

        self.synchronous_data = synchronous_data

        # Setting orbit length
        if orbit_bump is None:
            self.length = ring_programs.orbit_length_program(
                self.length_design)
        else:
            if not isinstance(orbit_bump, ring_programs.orbit_length_program):
                orbit_bump = ring_programs.orbit_length_program(orbit_bump)
            if orbit_bump.timebase == 'by_time':
                orbit_bump[:, 1, :] += self.length_design
            else:
                orbit_bump += self.length_design
            self._check_and_set_alpha_and_orbit(orbit_bump)

        # Setting the linear momentum compaction factor
        # Checking that the synchronous data and the momentum compaction
        # have the same length if defined turn-by-turn, raise a warning if one
        # is defined by turn and the other time based
        self.alpha_orders = [0]
        alpha_order_max = 0
        if not isinstance(alpha_0, ring_programs.momentum_compaction):
            alpha_0 = ring_programs.momentum_compaction(alpha_0, order=0)
        else:
            if alpha_0.order != 0:
                raise excpt.InputError(
                    "The order of the datatype passed as keyword " +
                    "argument alpha_%s do not match" % (0))

        self._check_and_set_alpha_and_orbit(alpha_0, 0)

        # Treating non-linear momentum compaction factor if declared
        # Listing all the declared alpha first
        alpha_n = {1: alpha_1, 2: alpha_2}

        if alpha_1 is not None:
            alpha_order_max = 1
        if alpha_2 is not None:
            alpha_order_max = 2

        for argument in kwargs:
            if 'alpha' in argument:
                try:
                    order = int(argument.split('_')[-1])
                    alpha_order_max = np.max([alpha_order_max, order])
                    alpha_n[order] = kwargs[argument]
                except Exception:
                    raise excpt.InputError(
                        'The keyword argument ' + argument + ' was ' +
                        'interpreted as non-linear momentum compaction ' +
                        'factor. ' +
                        'The correct syntax is alpha_n.')

        # Setting all the valid non-linear alpha and replacing
        # undeclared orders with zeros
        for order in range(1, alpha_order_max + 1):
            alpha = alpha_n.pop(order, None)

            if alpha is None:
                alpha = 0
                # This condition can be replaced by 'continue' to avoid
                # populating the object with 0 programs
            else:
                self.alpha_orders.append(order)

            if not isinstance(alpha, ring_programs.momentum_compaction):
                alpha = ring_programs.momentum_compaction(alpha, order=order)
            else:
                if alpha.order != order:
                    raise excpt.InputError(
                        "The order of the datatype passed as keyword " +
                        "argument alpha_%s do not match" % (order))

            self._check_and_set_alpha_and_orbit(alpha, order)

    def _check_and_set_alpha_and_orbit(self, alpha_or_orbit, order=None):
        '''
        Internal function to check that the input momentum compaction or orbit
        length is coherent with the synchronous data. If the synchronous data
        is turn based, the momentum compaction or orbit should have the same
        length.
        If the synchronous is time based while the momentum compaction or
        orbit is turn based, raises a warning.
        '''

        # attr_name is the attribute to apply to RingSection
        # attr_name_err is for the warning message
        if order is not None:
            attr_name = 'alpha_' + str(order)
            attr_name_err = attr_name
        else:
            attr_name = 'length'
            attr_name_err = 'orbit_bump'

        setattr(self, attr_name, alpha_or_orbit)

        if (self.synchronous_data.timebase == 'single') and \
                (alpha_or_orbit.timebase != 'single'):

            warn_message = 'The synchronous data was defined as single element while the ' + \
                'input ' + attr_name_err + ' was defined turn or time based. ' + \
                'Only the first element of the program will be taken in ' + \
                'the Ring object after treatment.'
            warnings.warn(warn_message)

        if (self.synchronous_data.timebase == 'by_turn') and \
                (alpha_or_orbit.timebase == 'by_turn'):

            if (alpha_or_orbit.shape[-1] > 1) and \
                    (self.synchronous_data.shape[-1]
                     > alpha_or_orbit.shape[-1]):

                raise excpt.InputError(
                    'The input ' + attr_name_err +
                    ' was passed as a turn based program but with ' +
                    'different length than the synchronous data. ' +
                    'Turn based programs should have the same length.')

        elif (self.synchronous_data.timebase == 'by_time') and \
                (alpha_or_orbit.timebase == 'by_turn'):

            warn_message = 'The synchronous data was defined time based while the ' + \
                'input ' + attr_name_err + ' was defined turn base, this may' + \
                'lead to errors in the Ring object after interpolation.'
            warnings.warn(warn_message)
