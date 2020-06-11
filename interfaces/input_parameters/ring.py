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
    :Authors: **Simon Albright**, **Alexandre Lasheen**, **Danilo Quartullo**, **Helga Timko**
'''

# General imports
import numpy as np
import warnings
from scipy.constants import c

# BLonD_Common imports
from ...devtools import exceptions as excpt
from ...devtools import assertions as assrt
from ..beam import beam
from ...datatypes import ring_programs
from ...datatypes.blond_function import machine_program
from ...utilities import timing as tmng
from ...utilities import rel_transforms as rt
from ...maths import calculus as calc
from .ring_section import RingSection


class Ring:
    r""" Class containing the general properties of the synchrotron that are
    independent of the RF system or the beam.

    The index :math:`n` denotes time steps, :math:`k` ring segments/sections
    and :math:`i` momentum compaction orders.

    Parameters
    ----------
    Particle : class
        A Particle-based class defining the primary, synchronous particle (mass
        and charge) that is reference for the momentum/energy in the ring.
    Section_list : list
        The list of all the sections to build the Ring.

    Attributes
    ----------
    circumference : float
        Circumference of the synchrotron. Sum of ring segment lengths,
        :math:`C = \sum_k L_k` [m]
    radius : float
        Radius of the synchrotron, :math:`R = C/(2 \pi)` [m]
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
    t_rev : float array [n_turns+1]
        Revolution period turn by turn.
        :math:`T_{0,n} = \frac{C}{\beta_{s,n} c}` [s]
    f_rev : float array [n_turns+1]
        Revolution frequency :math:`f_{0,n} = \frac{1}{T_{0,n}}` [Hz]
    omega_rev : float array [n_turns+1]
        Revolution angular frequency :math:`\omega_{0,n} = 2\pi f_{0,n}` [1/s]
    cycle_time : float array [n_turns+1]
        Cumulative cycle time, turn by turn, :math:`t_n = \sum_n T_{0,n}` [s].
        Possibility to extract cycle parameters at these moments using
        'parameters_at_time'.
    alpha_order : int
        Highest order of momentum compaction (as defined by the input). Can
        be 0,1,2.
    RingOptions : RingOptions()
        The RingOptions is kept as an attribute of the Ring object for further
        usage.

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
    >>>
    >>>
    >>> # To declare a two section synchrotron at constant energy and
    >>> # higher-order momentum compaction factors; particle Electron:
    >>> from beam.beam import Electron
    >>> from input_parameters.ring import Ring
    >>>
    >>> n_turns = 10
    >>> C = [13000, 13659]
    >>> alpha_0 = [[3.21e-4], [2.89e-4]]  # or [3.21e-4, 2.89e-4]
    >>> alpha_1 = [[2.e-5], [1.e-5]]  # or [2.e-5, 1.e-5]
    >>> alpha_2 = [[5.e-7], [5.e-7]]  # or [5.e-7, 5.e-7]
    >>> momentum = 450e9
    >>> ring = Ring(C, alpha_0, momentum, Electron(), n_turns,
    >>>             alpha_1=alpha_1, alpha_2=alpha_2)

    """

    def __init__(self, Particle, Section_list, **kwargs):

        # Primary particle mass and charge used for energy calculations
        # If a string is passed, will generate the relevant Particle object
        # based on the name
        if isinstance(Particle, beam.Particle):
            self.Particle = Particle
        elif isinstance(Particle, str):
            self.Particle = beam.make_particle(Particle)
        else:
            raise excpt.InputError("The input Particle should be a " +
                                   "Particle object or a str.")

        # Getting all sections and checking their types
        if isinstance(Section_list, RingSection):
            Section_list = [Section_list]
        if not isinstance(Section_list, list):
            raise excpt.InputError("The Section_list should be a list or " +
                                   "a single RingSection object instance.")
        else:
            for section in Section_list:
                if not isinstance(section, RingSection):
                    raise excpt.InputError(
                        "The Section_list should be exclusively composed " +
                        "of RingSection object instances.")

                # Setting the Ring as attribute of the sections for reference
                section._Ring = self

        self.Section_list = Section_list
        self.n_sections = len(self.Section_list)

        # Extracting the length of sections to get circumference
        self.section_length = np.array(
            [section.length for section in self.Section_list])
        self.circumference = np.sum(self.section_length)

        # Computing ring radius
        self.radius = self.circumference / (2 * np.pi)

        # Extracting the bending radius from all sections
        self.bending_radius = np.array([
            section.bending_radius for section in self.Section_list])

        # Extracting the synchronous data from the sections
        for index_section, section in enumerate(self.Section_list):
            if index_section == 0:
                self.synchronous_data = section.synchronous_data
            else:
                if (section.synchronous_data == self.synchronous_data).all:
                    pass
                else:
                    warn_message = 'The synchronous data for all sections ' + \
                        'is not identical. This case is not yet fully ' + \
                        'tested and is under implementation.'
                    warnings.warn(warn_message)

        # Processing the momentum program
        # Getting the options to get at which time samples the interpolation
        # is made
        t_start = kwargs.pop('t_start', 0)
        t_stop = kwargs.pop('t_stop', np.inf)
        interp_time = kwargs.pop('interp_time', 0)

        # Treating the input of the interp_time option
        if not hasattr(interp_time, '__iter__'):
            interp_time = (interp_time, )

        # Getting the sample times
        sample_func, start, stop = tmng.time_from_sampling(*interp_time)

        # Keeping only the values between the user defined bounds
        if t_start > start:
            start = t_start
        if t_stop < stop:
            stop = t_stop

        # Getting options for the interpolation
        interpolation = kwargs.pop('interpolation', 'linear')
        store_turns = kwargs.pop('store_turns', True)

        # Processing the momentum program and interpolating on the
        # values defined by sample_func
        momentum_processed = self.synchronous_data.preprocess(
            self.Particle.mass,
            self.circumference, sample_func,
            interpolation, start, stop,
            store_turns=store_turns)

        # Getting the cycle time from the interpolation
        self.cycle_time = momentum_processed[1]

        # The machine turn numbers corresponding to the cycle_time
        # are kept if the store_turns is enabled
        if store_turns:
            self.parameters_at_turn = self._parameters_at_turn
            self.use_turns = momentum_processed[0].astype(int)
        else:
            self.parameters_at_turn = self._no_parameters_at_turn
            self.use_turns = momentum_processed[0]

        # Updating the number of turns in case it was changed after ramp
        # interpolation
        self.n_turns = momentum_processed.n_turns

        # Generating the attributes where to store the synchronous data
        # and associated parameters
        program_shape = (self.n_sections, self.n_turns)
        self.momentum = ring_programs.momentum_program.zeros(program_shape)
        self.energy = ring_programs.total_energy_program.zeros(program_shape)
        self.kin_energy = ring_programs.kinetic_energy_program.zeros(
            program_shape)
        self.beta = ring_programs._synchronous_data_program.zeros(
            program_shape)
        self.gamma = ring_programs._synchronous_data_program.zeros(
            program_shape)

        # Populating the synchronous data attribute each row corresponding
        # to a section
        for index_section in range(self.n_sections):
            self.momentum[index_section, :] = momentum_processed[2:]
            self.beta[index_section, :] = rt.mom_to_beta(
                momentum_processed[2:], self.Particle.mass)
            self.gamma[index_section, :] = rt.mom_to_gamma(
                momentum_processed[2:], self.Particle.mass)
            self.energy[index_section, :] = rt.momentum_to_energy(
                momentum_processed[2:], self.Particle.mass)
            self.kin_energy[index_section, :] = rt.momentum_to_kin_energy(
                momentum_processed[2:], self.Particle.mass)

        # Extracting and combining the orbit length programs
        self.orbit_length = ring_programs.orbit_length_program.combine_single_sections(
            *[section.orbit_length for section in self.Section_list])

        # Reshaping to match the dimensions of the synchronous data program
        self.orbit_length = self.orbit_length.reshape(
            self.n_sections, self.cycle_time, self.use_turns)

        # Computing the revolution period on the design orbit
        # as well as revolution frequency and angular frequency
        self.t_rev = np.dot(self.section_length, 1 / (self.beta * c))
        self.f_rev = 1 / self.t_rev
        self.omega_rev = 2 * np.pi * self.f_rev

        # Computing the time of flight in each section
        # and the revolution period on the beam orbit including
        # possible orbit bumps
        self.tof_section_orbit = np.array(self.orbit_length / (self.beta * c))
        self.t_rev_orbit = np.sum(self.tof_section_orbit, axis=0)
        self.f_rev_orbit = 1 / self.t_rev_orbit
        self.omega_rev_orbit = 2 * np.pi * self.f_rev_orbit

        # Recalculating the delta_E
        self.delta_E = np.diff(self.energy, axis=1)
        if self.n_turns > len(self.use_turns):
            self.delta_E = np.zeros(self.energy.shape)
            self._recalc_delta_E()

        # Determining the momentum compaction orders defined in all sections
        # The orders 1 and 2 are presently set by default to zeros if
        # not defined in sections for the calculation of all orders of eta
        self.alpha_orders_defined = [
            section.alpha_orders_defined
            for section in self.Section_list] + [[1], [2]]

        self.alpha_orders_defined = np.unique(self.alpha_orders_defined)
        self.alpha_order = np.max(self.alpha_orders_defined)

        # Getting the momentum compaction programs in all sections
        # and combining the programs (the missing programs are filled
        # with zeros).
        # The programs are reshaped to the size of the momentum program
        for alpha_order in self.alpha_orders_defined:
            alpha_prog = []
            alpha_name = 'alpha_%d' % (alpha_order)
            for section in self.Section_list:
                if hasattr(section, alpha_name):
                    alpha_prog.append(getattr(section, alpha_name))
                else:
                    alpha_prog.append(ring_programs.momentum_compaction(
                        0, n_turns=self.n_turns))
                    alpha_prog[-1].order = alpha_order
                    alpha_prog[-1].timebase = 'by_turn'
            print(alpha_order, alpha_prog)
            alpha_prog = ring_programs.momentum_compaction.combine_single_sections(
                *alpha_prog)
            setattr(self, alpha_name, alpha_prog.reshape(
                self.n_sections, self.cycle_time, self.use_turns))

        # Slippage factor derived from alpha, beta, gamma
        for order in range(3):
            setattr(self, 'eta_%d' % (order), np.zeros(self.momentum.shape))
        self._eta_generation()

#     @classmethod
#     def direct_input(self, section_length, alpha, Particle,
#                      momentum=None, kin_energy=None, energy=None,
#                      bending_field=None, bending_radius=None,
#                      **kwargs):
# 
#         # Checking that at least one synchronous data input is passed
#         syncDataTypes = ('momentum', 'kin_energy', 'energy', 'B_field')
#         syncDataInput = (momentum, kin_energy, energy, bending_field)
#         assrt.single_not_none(*syncDataInput,
#                               msg='Exactly one of ' + str(syncDataTypes) +
#                               ' must be declared',
#                               exception=excpt.InputError)
# 
#         # Checking that the bending_radius is passed with the bending_field
#         if bending_field is not None and bending_radius is None:
#             raise excpt.InputError("If bending_field is used, bending_radius "
#                                    + "must be defined.")
# 
#         # Taking the first synchronous_data input not declared as None
#         # The assertion above ensures that only one is declared
#         for func_type, synchronous_data in zip(syncDataTypes, syncDataInput):
#             if synchronous_data is not None:
#                 break
# 
#         # Setting ring length, circumerence, radius, bending radius if defined
#         self.section_length = np.array(section_length, ndmin=1, dtype=float)
#         self.circumference = np.sum(self.section_length)
#         self.radius = self.circumference / (2 * np.pi)
# 
#         if bending_radius is not None:
#             self.bending_radius = float(bending_radius)
#         else:
#             self.bending_radius = bending_radius
# 
#         # Primary particle mass and charge used for energy calculations
#         # If a string is passed, will generate the relevant Particle object
#         # based on the name
#         if isinstance(Particle, beam.Particle):
#             self.Particle = Particle
#         else:
#             self.Particle = beam.make_particle(Particle)
# 
#         # Reshaping the input synchronous data to the adequate format and
#         # get back the momentum program from RingOptions
#         if not isinstance(synchronous_data, dTypes._ring_program):
#             synchronous_data = \
#                 dTypes._ring_program.conversions[func_type](synchronous_data)
# 
#         if synchronous_data.shape[0] != len(self.section_length):
#             raise excpt.InputDataError("ERROR in Ring: Number of sections " +
#                                        "and ring length size do not match!")
# 
#         t_start = kwargs.pop('t_start', 0)
#         t_stop = kwargs.pop('t_stop', np.inf)
#         interp_time = kwargs.pop('interp_time', 0)
# 
#         if not hasattr(interp_time, '__iter__'):
#             interp_time = (interp_time, )
# 
#         sample_func, start, stop = tmng.time_from_sampling(*interp_time)
# 
#         if t_start > start:
#             start = t_start
#         if t_stop < stop:
#             stop = t_stop
# 
#         synchronous_data.convert(self.Particle.mass, self.Particle.charge,
#                                  self.bending_radius)
# 
#         interpolation = kwargs.pop("interpolation", 'linear')
#         store_turns = kwargs.pop('store_turns', True)
# 
#         self.momentum = synchronous_data.preprocess(
#             self.Particle.mass,
#             self.circumference, sample_func,
#             interpolation, start, stop,
#             store_turns=store_turns)
# 
#         self.n_sections = self.momentum.shape[0] - 2
#         self.cycle_time = self.momentum[1]
#         if store_turns:
#             self.parameters_at_turn = self._parameters_at_turn
#             self.use_turns = self.momentum[0].astype(int)
#         else:
#             self.parameters_at_turn = self._no_parameters_at_turn
#             self.use_turns = self.momentum[0]
#         # Updating the number of turns in case it was changed after ramp
#         # interpolation
#         self.n_turns = self.momentum.n_turns
# 
#         # Derived from momentum
#         self.beta = rt.mom_to_beta(self.momentum[2:], self.Particle.mass)
#         self.gamma = rt.mom_to_gamma(self.momentum[2:], self.Particle.mass)
#         self.energy = rt.momentum_to_energy(self.momentum[2:],
#                                             self.Particle.mass)
#         self.kin_energy = rt.momentum_to_kin_energy(self.momentum[2:],
#                                                     self.Particle.mass)
#         self.t_rev = np.dot(self.section_length, 1 / (self.beta * c))
#         self.delta_E = np.diff(self.energy, axis=1)
#         if self.n_turns > len(self.use_turns):
#             self.delta_E = np.zeros(self.energy.shape)
#             self._recalc_delta_E()
# 
#         self.momentum = self.momentum[2:]
# 
#         self.f_rev = 1 / self.t_rev
#         self.omega_rev = 2 * np.pi * self.f_rev
# 
#         # Momentum compaction, checks, and derived slippage factors
# 
#         if not hasattr(alpha, '__iter__'):
#             alpha = (alpha, )
# 
#         if isinstance(alpha, dict):
#             try:
#                 if not all([k % 1 == 0 for k in alpha.keys()]):
#                     raise TypeError
#             except TypeError:
#                 raise excpt.InputError("If alpha is dict all keys must be "
#                                        + "numeric and integer")
# 
#             maxAlpha = np.max(tuple(alpha.keys())).astype(int)
#             alpha = [alpha.pop(i, 0) for i in range(maxAlpha + 1)]
# 
#         if isinstance(alpha, dTypes._function):
#             alpha = (alpha,)
# 
#         for i, a in enumerate(alpha):
#             if not isinstance(a, dTypes.momentum_compaction):
#                 a = dTypes.momentum_compaction(a, order=i)
# 
#             setattr(self, 'alpha_' + str(i), a.reshape(self.n_sections,
#                                                        self.cycle_time,
#                                                        self.use_turns))
#             setattr(self, 'eta_' + str(i), np.zeros([self.n_sections,
#                                                      len(self.use_turns)]))
#         self.alpha_order = i
# 
#         for i in range(3 - self.alpha_order):
#             if not hasattr(self, f'alpha_{i}'):
#                 setattr(self, 'alpha_' + str(i), np.zeros([self.n_sections,
#                                                            len(self.use_turns)]))
#                 setattr(self, 'eta_' + str(i), np.zeros([self.n_sections,
#                                                          len(self.use_turns)]))

    def _eta_generation(self):
        """ Function to generate the slippage factors (zeroth, first, and
        second orders, see [1]_) from the momentum compaction and the
        relativistic beta and gamma program through the cycle.

        References
        ----------
        .. [1] "Accelerator Physics," S. Y. Lee, World Scientific,
                Third Edition, 2012.
        """

        for i in range(np.min([self.alpha_order + 1, 3])):
            getattr(self, '_eta' + str(i))()

    def _eta0(self):
        """ Function to calculate the zeroth order slippage factor eta_0 """

        for i in range(0, self.n_sections):
            self.eta_0[i] = self.alpha_0[i] - self.gamma[i]**(-2.)

    def _eta1(self):
        """ Function to calculate the first order slippage factor eta_1 """

        for i in range(0, self.n_sections):
            self.eta_1[i] = 3 * self.beta[i]**2 / (2 * self.gamma[i]**2) + \
                self.alpha_1[i] - self.alpha_0[i] * self.eta_0[i]

    def _eta2(self):
        """ Function to calculate the second order slippage factor eta_2 """

        for i in range(0, self.n_sections):
            self.eta_2[i] = - self.beta[i]**2 * (5 * self.beta[i]**2 - 1) / \
                (2 * self.gamma[i]**2) + self.alpha_2[i] - 2 * self.alpha_0[i] *\
                self.alpha_1[i] + self.alpha_1[i] / self.gamma[i]**2 + \
                self.alpha_0[i]**2 * self.eta_0[i] - 3 * self.beta[i]**2 * \
                self.alpha_0[i] / (2 * self.gamma[i]**2)

    def parameters_at_time(self, cycle_moments):
        """ Function to return various cycle parameters at a specific moment in
        time. The cycle time is defined to start at zero in turn zero.

        Parameters
        ----------
        cycle_moments : float array
            Moments of time at which cycle parameters are to be calculated [s].

        Returns
        -------
        parameters : dictionary
            Contains 'momentum', 'beta', 'gamma', 'energy', 'kin_energy',
            'f_rev', 't_rev'. 'omega_rev', 'eta_0', and 'delta_E' interpolated
            to the moments contained in the 'cycle_moments' array

        """

        parameters = {}
        parameters['momentum'] = np.interp(cycle_moments, self.cycle_time,
                                           self.momentum[0])
        parameters['beta'] = np.interp(cycle_moments, self.cycle_time,
                                       self.beta[0])
        parameters['gamma'] = np.interp(cycle_moments, self.cycle_time,
                                        self.gamma[0])
        parameters['energy'] = np.interp(cycle_moments, self.cycle_time,
                                         self.energy[0])
        parameters['kin_energy'] = np.interp(cycle_moments, self.cycle_time,
                                             self.kin_energy[0])
        parameters['f_rev'] = np.interp(cycle_moments, self.cycle_time,
                                        self.f_rev)
        parameters['t_rev'] = np.interp(cycle_moments, self.cycle_time,
                                        self.t_rev)
        parameters['omega_rev'] = np.interp(cycle_moments, self.cycle_time,
                                            self.omega_rev)
        parameters['eta_0'] = np.interp(cycle_moments, self.cycle_time,
                                        self.eta_0[0])
        if len(self.cycle_time) == len(self.delta_E[0]):
            parameters['delta_E'] = np.interp(cycle_moments,
                                              self.cycle_time,
                                              self.delta_E[0])
        else:
            parameters['delta_E'] = np.interp(cycle_moments,
                                              self.cycle_time[:-1],
                                              self.delta_E[0])
        parameters['charge'] = self.Particle.charge
        parameters['cycle_time'] = cycle_moments

        return parameters

    def _no_parameters_at_turn(self, turn):
        raise RuntimeError("parameters_at_turn only available if " +
                           "store_turns = True at object declaration")

    def _parameters_at_turn(self, turn):

        try:
            sample = np.where(self.use_turns == turn)[0][0]
        except IndexError:
            raise excpt.InputError("turn " + str(turn) + " has not been "
                                   + "stored for the specified interpolation")
        else:
            return self.parameters_at_sample(sample)

    def parameters_at_sample(self, sample):

        parameters = {}
        parameters['momentum'] = self.momentum[0, sample]
        parameters['beta'] = self.beta[0, sample]
        parameters['gamma'] = self.gamma[0, sample]
        parameters['energy'] = self.energy[0, sample]
        parameters['kin_energy'] = self.kin_energy[0, sample]
        parameters['f_rev'] = self.f_rev[sample]
        parameters['t_rev'] = self.t_rev[sample]
        parameters['omega_rev'] = self.omega_rev[sample]
        parameters['eta_0'] = self.eta_0[0, sample]
        try:
            parameters['delta_E'] = self.delta_E[0, sample]
        except IndexError:
            parameters['delta_E'] = self.delta_E[0, sample - 1]
        parameters['charge'] = self.Particle.charge
        parameters['cycle_time'] = self.cycle_time[sample]

        return parameters

    def _recalc_delta_E(self):
        """
        Function to recalculate delta_E.
        If interpolation is not done on a turn-by-turn basis the delta_E will
        not be correct.  This function recalculates it to give the correct
        value for each turn.
        """

        for section in range(self.n_sections):
            derivative = calc.deriv_cubic(self.cycle_time,
                                          self.energy[section])[1]
            self.delta_E[section] = derivative * self.t_rev
