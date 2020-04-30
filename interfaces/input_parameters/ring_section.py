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
from ...datatypes import datatypes as dTypes
from ...datatypes.blond_function import machine_program
from ...devtools import exceptions as excpt
from ...devtools import assertions as assrt


class Section:
    r""" Class containing the general properties of a section of the
    accelerator that are independent of the RF system or the beam.

    Parameters
    ----------
    section_length : float (opt: list or np.ndarray)
        Length [m] accelerator section;
        can be input as single float or as a program (1D array is a
        turn-by-turn program and 2D array is a time dependent program).
    alpha_0 : float (opt: list or np.ndarray)
        Momentum compaction factor of zeroth order :math:`\alpha_{0,k,i}` [1];
        can be input as single float or as a program (1D array is a
        turn-by-turn program and 2D array is a time dependent program).
    synchronous_data : float (opt: list or np.ndarray)
        Design synchronous particle momentum (default) [eV], kinetic or
        total energy [eV] or bending field [T] on the design orbit.
        Input for each RF section :math:`p_{s,k,n}`.
        Can be input as a single constant float, or as a
        program of (n_turns + 1) turns. In case of several sections without
        acceleration, input: [[momentum_section_1], [momentum_section_2],
        etc.]. In case of several sections with acceleration, input:
        [momentum_program_section_1, momentum_program_section_2, etc.]. Can
        be input also as a tuple of time and momentum, see also
        'cycle_time' and 'PreprocessRamp'
    bending_radius : float (opt: list or np.ndarray)
        Optional: Radius [m] of the bending magnets,
        required if 'bending field' is set for the synchronous_data_type
    alpha_1 : float (opt: list or np.ndarray)
        Optional : Momentum compaction factor of first order
        :math:`\alpha_{1,k,i}` [1]; can be input as single float or as a
        program of (n_turns + 1) turns (should be of the same size as
        synchronous_data and alpha_0).
    alpha_2 : float (opt: list or np.ndarray)
        Optional : Momentum compaction factor of second order
        :math:`\alpha_{2,k,i}` [1]; can be input as single float or as a
        program of (n_turns + 1) turns (should be of the same size as
        synchronous_data and alpha_0).
    alpha_n : float (opt: list or np.ndarray)
        Optional : Higher order momentum compaction can also be passed through
        extra keyword arguments.

    Attributes
    ----------
    section_length : float
        Circumference of the synchrotron. Sum of ring segment lengths,
        :math:`C = \sum_k L_k` [m]
    bending_radius : float
        Bending radius in dipole magnets, :math:`\rho` [m]
    alpha_order : int
        Number of orders of the momentum compaction factor (from 0 to 2)
    eta_0 : float matrix [n_sections, n_turns+1]
        Zeroth order slippage factor :math:`\eta_{0,k,n} = \alpha_{0,k,n} -
        \frac{1}{\gamma_{s,k,n}^2}` [1]
    eta_1 : float matrix [n_sections, n_turns+1]
        First order slippage factor :math:`\eta_{1,k,n} =
        \frac{3\beta_{s,k,n}^2}{2\gamma_{s,k,n}^2} + \alpha_{1,k,n} -
        \alpha_{0,k,n}\eta_{0,k,n}` [1]
    eta_2 : float matrix [n_sections, n_turns+1]
        Second order slippage factor :math:`\eta_{2,k,n} =
        -\frac{\beta_{s,k,n}^2\left(5\beta_{s,k,n}^2-1\right)}
        {2\gamma_{s,k,n}^2} + \alpha_{2,k,n} - 2\alpha_{0,k,n}\alpha_{1,k,n}
        + \frac{\alpha_{1,k,n}}{\gamma_{s,k,n}^2} + \alpha_{0,k}^2\eta_{0,k,n}
        - \frac{3\beta_{s,k,n}^2\alpha_{0,k,n}}{2\gamma_{s,k,n}^2}` [1]
    momentum : float matrix [n_sections, n_turns+1]
        Synchronous relativistic momentum on the design orbit :math:`p_{s,k,n}`
    beta : float matrix [n_sections, n_turns+1]
        Synchronous relativistic beta program for each segment of the
        ring :math:`\beta_{s,k}^n = \frac{1}{\sqrt{1
        + \left(\frac{m}{p_{s,k,n}}\right)^2} }` [1]
    gamma : float matrix [n_sections, n_turns+1]
        Synchronous relativistic gamma program for each segment of the ring
        :math:`\gamma_{s,k,n} = \sqrt{ 1
        + \left(\frac{p_{s,k,n}}{m}\right)^2 }` [1]
    energy : float matrix [n_sections, n_turns+1]
        Synchronous total energy program for each segment of the ring
        :math:`E_{s,k,n} = \sqrt{ p_{s,k,n}^2 + m^2 }` [eV]
    kin_energy : float matrix [n_sections, n_turns+1]
        Synchronous kinetic energy program for each segment of the ring
        :math:`E_{s,kin} = \sqrt{ p_{s,k,n}^2 + m^2 } - m` [eV]
    delta_E : float matrix [n_sections, n_turns]
        Gain in synchronous total energy from one point to another,
        for all sections,
        :math:`: \quad E_{s,k,n+1}- E_{s,k,n}` [eV]
    cycle_time : float array [n_turns+1]
        Cumulative cycle time, turn by turn, :math:`t_n = \sum_n T_{0,n}` [s].
        Possibility to extract cycle parameters at these moments using
        'parameters_at_time'.
    alpha_order : int
        Highest order of momentum compaction (as defined by the input). Can
        be 0,1,2.

    Examples
    --------
    >>> # To declare a single-section synchrotron at constant energy:
    >>> # Particle type Proton
    >>> from beam.beam import Proton
    >>> from input_parameters.ring import Ring
    >>>
    >>> n_turns = 10
    >>> C = 26659
    >>> alpha_0 = 3.21e-4
    >>> momentum = 450e9
    >>> ring = Ring(C, alpha_0, momentum, Proton(), n_turns)

    """

    def __init__(self, section_length, alpha_0,
                 momentum=None, kin_energy=None, energy=None,
                 bending_field=None, bending_radius=None,
                 alpha_1=None, alpha_2=None, **kwargs):

        # Setting section length
        self.section_length = dTypes._ring_function(section_length)

        # Checking that at least one synchronous data input is passed
        syncDataTypes = ('momentum', 'kin_energy', 'energy', 'B_field')
        syncDataInput = (momentum, kin_energy, energy, bending_field)
        assrt.single_not_none(*syncDataInput,
                              msg='Exactly one of '+str(syncDataTypes) +
                              ' must be declared',
                              exception=excpt.InputError)

        # Checking that the bending_radius is passed with the bending_field
        # and setting the bending_radius if defined
        self.bending_radius = None
        if bending_field is not None and bending_radius is None:
            raise excpt.InputError("If bending_field is used, " +
                                   "bending_radius must be defined.")
        else:
            self.bending_radius = dTypes._ring_function(bending_radius)

        # Taking the first synchronous_data input not declared as None
        # The assertion above ensures that only one is declared
        for func_type, synchronous_data in zip(syncDataTypes, syncDataInput):
            if synchronous_data is not None:
                break

        # Reshaping the input synchronous data to the adequate format and
        # get back the momentum program
        if not isinstance(synchronous_data, dTypes._ring_program):
            synchronous_data \
                = dTypes._ring_program.conversions[func_type](synchronous_data)

        self.synchronous_data = synchronous_data

        # Setting the linear momentum compaction factor
        # Checking that the synchronous data and the momentum compaction
        # have the same length if defined turn-by-turn, raise a warning if one
        # is defined by turn and the other time based
        self.alpha_order = 0
        if not isinstance(alpha_0, dTypes.momentum_compaction):
            alpha_0 = dTypes.momentum_compaction(alpha_0, order=0)
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
                        'The keyword argument '+argument+' was interpreted ' +
                        'as non-linear momentum compaction factor. ' +
                        'The correct syntax is alpha_n.')

        # Setting all the valid non-linear alpha and replacing
        # undeclared orders with zeros
        self.alpha_orders_defined = [0]
        for order in range(1, self.alpha_order+1):
            alpha = alpha_n.pop(order, None)

            if alpha is None:
                alpha = 0
                # This condition can be replaced by 'continue' to avoid
                # populating the object with 0 programs
            else:
                self.alpha_orders_defined.append(order)

            if not isinstance(alpha, dTypes.momentum_compaction):
                alpha = dTypes.momentum_compaction(alpha, order=order)
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

        setattr(self, 'alpha_'+str(order), alpha)

        if (self.synchronous_data.timebase == 'by_turn') and \
                (alpha.timebase == 'by_turn'):

            if (alpha.shape[-1] > 1) and \
                    (self.synchronous_data.shape[-1] > alpha.shape[-1]):

                raise excpt.InputError(
                            'The momentum compaction alpha_'+str(order) +
                            ' was passed as a turn based program but with ' +
                            'different length than the synchronous data. ' +
                            'Turn based programs should have the same length.')

        elif (self.synchronous_data.timebase == 'by_time') and \
                (alpha.timebase == 'by_turn'):

            warn_message = 'The synchronous data was defined time based while the ' + \
                'momentum compaction was defined turn base, this may' + \
                'lead to errors in the Ring object after interpolation'
            warnings.warn(warn_message)
