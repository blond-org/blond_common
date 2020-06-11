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
    length : float (opt: list or np.ndarray)
        Length [m] accelerator section;
        can be input as single float or as a program (1D array is a
        turn-by-turn program and 2D array is a time dependent program).
        If a turn-by-turn program is passed, should be of the same size
        as the synchronous data.
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
    alpha_1 : float (opt: list or np.ndarray)
        Optional : Momentum compaction factor of first order
        :math:`\alpha_{1}`;
        can be input as single float or as a program (1D array is a
        turn-by-turn program and 2D array is a time dependent program).
        If a turn-by-turn program is passed, should be of the same size
        as the synchronous data.
    alpha_2 : float (opt: list or np.ndarray)
        Optional : Momentum compaction factor of second order
        :math:`\alpha_{2}`;
        can be input as single float or as a program (1D array is a
        turn-by-turn program and 2D array is a time dependent program).
        If a turn-by-turn program is passed, should be of the same size
        as the synchronous data.
    alpha_n : float (opt: list or np.ndarray)
        Optional : Higher order momentum compaction can also be passed through
        extra keyword arguments;
        can be input as single float or as a program (1D array is a
        turn-by-turn program and 2D array is a time dependent program).
        If a turn-by-turn program is passed, should be of the same size
        as the synchronous data.

    Attributes
    ----------
    length : datatype.length_function
        Length of the section [m]
    synchronous_data : datatype._ring_program
        The user input synchronous data, with no conversion applied.
        The datatype depends on the user input and can be
        momentum_program, kinetic_energy_program, total_energy_program,
        bending_field_program
    bending_radius : float (or None)
        Bending radius in dipole magnets, :math:`\rho` [m]
    alpha_0 : datatype.momentum_compaction
        Momentum compaction factor of zeroth order
    alpha_1 : datatype.momentum_compaction (or None)
        Momentum compaction factor of first order
    alpha_2 : datatype.momentum_compaction (or None)
        Momentum compaction factor of second order
    alpha_n : datatype.momentum_compaction (or undefined)
        Momentum compaction factor of higer orders
    alpha_order : int
        Maximum order of momentum compaction
    alpha_orders_defined : int
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
    >>> alpha_0 = 1e-3
    >>> alpha_1 = 1e-4
    >>> alpha_2 = 1e-5
    >>> alpha_5 = 1e-9
    >>> energy = machine_program([[0, 1, 2],
    >>>                           [26e9, 27e9, 28e9]])
    >>>
    >>> section = RingSection(length, alpha_0, energy=energy,
    >>>                       alpha_1=alpha_1, alpha_2=alpha_2,
    >>>                       alpha_5=alpha_5)
    """

    def __init__(self, length, alpha_0,
                 momentum=None, kin_energy=None, energy=None,
                 bending_field=None, bending_radius=None,
                 alpha_1=None, alpha_2=None, **kwargs):

        # Setting section length
        self.length = ring_programs.length_function(length)

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

        # Setting the linear momentum compaction factor
        # Checking that the synchronous data and the momentum compaction
        # have the same length if defined turn-by-turn, raise a warning if one
        # is defined by turn and the other time based
        self.alpha_order = 0
        if not isinstance(alpha_0, ring_programs.momentum_compaction):
            alpha_0 = ring_programs.momentum_compaction(alpha_0, order=0)
        else:
            if alpha_0.order != 0:
                raise excpt.InputError(
                    "The order of the datatype passed as keyword " +
                    "argument alpha_%s do not match" % (0))

        self._check_and_set_momentum_compaction(alpha_0, 0)

        # Treating non-linear momentum compaction factor if declared
        # Listing all the declared alpha first
        alpha_n = {1: alpha_1, 2: alpha_2}

        if alpha_1 is not None:
            self.alpha_order = 1
        if alpha_2 is not None:
            self.alpha_order = 2

        for argument in kwargs:
            if 'alpha' in argument:
                try:
                    order = int(argument.split('_')[-1])
                    self.alpha_order = np.max([self.alpha_order, order])
                    alpha_n[order] = kwargs[argument]
                except Exception:
                    raise excpt.InputError(
                        'The keyword argument ' + argument + ' was ' +
                        'interpreted as non-linear momentum compaction ' +
                        'factor. ' +
                        'The correct syntax is alpha_n.')

        # Setting all the valid non-linear alpha and replacing
        # undeclared orders with zeros
        self.alpha_orders_defined = [0]
        for order in range(1, self.alpha_order + 1):
            alpha = alpha_n.pop(order, None)

            if alpha is None:
                alpha = 0
                # This condition can be replaced by 'continue' to avoid
                # populating the object with 0 programs
            else:
                self.alpha_orders_defined.append(order)

            if not isinstance(alpha, ring_programs.momentum_compaction):
                alpha = ring_programs.momentum_compaction(alpha, order=order)
            else:
                if alpha.order != order:
                    raise excpt.InputError(
                        "The order of the datatype passed as keyword " +
                        "argument alpha_%s do not match" % (order))

            self._check_and_set_momentum_compaction(alpha, order)

    def _check_and_set_momentum_compaction(self, alpha, order):
        '''
        Internal function to check that the input momentum compaction is
        coherent with the synchronous data. If the synchronous data is turn
        based, the momentum compaction should have the same length. If the
        synchronous is time based while the momentum compaction is turn based,
        raises a warning.
        '''

        setattr(self, 'alpha_' + str(order), alpha)

        if (self.synchronous_data.timebase == 'by_turn') and \
                (alpha.timebase == 'by_turn'):

            if (alpha.shape[-1] > 1) and \
                    (self.synchronous_data.shape[-1] > alpha.shape[-1]):

                raise excpt.InputError(
                    'The momentum compaction alpha_' + str(order) +
                    ' was passed as a turn based program but with ' +
                    'different length than the synchronous data. ' +
                    'Turn based programs should have the same length.')

        elif (self.synchronous_data.timebase == 'by_time') and \
                (alpha.timebase == 'by_turn'):

            warn_message = 'The synchronous data was defined time based while the ' + \
                'momentum compaction was defined turn base, this may' + \
                'lead to errors in the Ring object after interpolation'
            warnings.warn(warn_message)
