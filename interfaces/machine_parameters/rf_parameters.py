# coding: utf8
# Copyright 2014-2019 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Module gathering and processing all RF parameters used in the simulation.**

:Authors: **Alexandre Lasheen**, **Danilo Quartullo**, **Helga Timko**
'''

from __future__ import division, print_function
from builtins import str, range, object
import numpy as np
from scipy.constants import c
from scipy.integrate import cumtrapz
from ..beam.beam import Proton
import sys

#BLonD_Common imports
from ...datatypes import rf_programs as rfProgs
from ...devtools import exceptions as excpt
from ...devtools import assertions as assrt


class RFStation:
    r""" Class containing all the RF parameters for all the RF systems in one
    ring segment or RF station.

    **How to use RF programs:**

    * For 1 RF system and constant values of V, h, or phi, input a single value
    * For 1 RF system and varying values of V, h, or phi, input an array of
      n_turns+1 values
    * For several RF systems and constant values of V, h, or phi, input lists
      of single values
    * For several RF systems and varying values of V, h, or phi, input lists
      of arrays of n_turns+1 values
    * For pre-processing, pass a list of times-voltages, times-harmonics,
      and/or times-phases for **each** RF system as a tuple
      ((time_1, voltage_1), (time_2, voltage_2), ...)

    Optional: RF frequency other than the design frequency. In this case, need
    to use a beam phase loop for correct RF phase!

    Optional: empty RFStation (e.g. for machines with synchrotron radiation);
    input voltage as 0.

    The index :math:`n` denotes time steps, :math:`l` the index of the RF
    systems in the section.

    **N.B. for negative eta the RF phase has to be shifted by Pi w.r.t the time
    reference.**

    Parameters
    ----------
    Ring : class
        A Ring type class
    harmonic : float (opt: float array/matrix, tuple of float array/matrix)
        Harmonic number of the RF system, :math:`h_{l,n}` [1]. For input
        options, see above
    voltage : float (opt: float array/matrix, tuple of float array/matrix)
        RF cavity voltage as seen by the beam, :math:`V_{l,n}` [V]. For input
        options, see above
    phi_rf_d : float (opt: float array/matrix, tuple of float array/matrix)
        Programmed/designed RF cavity phase,
        :math:`\phi_{d,l,n}` [rad]. For input options, see above
    n_rf : int
        Optional, Number of harmonic rf systems in the section :math:`l`.
        Becomes mandatory for several rf systems.
    section_index : int
        Optional, In case of several sections in the Ring object, this
        specifies after which section the rf station is located (to get the
        right momentum program etc.). Value should be in the range
        1..Ring.n_sections
    omega_rf : float (opt: float array/matrix)
        Optional, Sets the rf angular frequency program that does not follow
        the harmonic condition. For input options, see above.
    phi_noise : float (opt: float array/matrix)
        Optional, programmed RF cavity phase noise, :math:`\phi_{N,l,n}` [rad].
        Added to all RF systems in the station. For input options, see above
    phi_modulation : class (opt: iterable of classes)
        A PhaseModulation type class (or iterable of classes)
    RFStationOptions : class
        Optionnal, A RFStationOptions-based class defining smoothing,
        interpolation, etc. options for harmonic, voltage, and/or
        phi_rf_d programme to be interpolated to a turn-by-turn programme

    Attributes
    ----------
    counter : int
        Counter of the current simulation time step; defined as a list in
        order to be passed by reference
    section_index : int
        Unique index :math:`k` of the RF station the present class is defined
        for. Input in the range 1..n_sections (see
        :py:class:`input_parameters.ring.Ring`).
        Inside the code, indices 0..n_sections-1 are used.
    Particle : class
        Inherited from
        :py:attr:`input_parameters.ring.Ring.Particle`
    n_turns : int
        Inherited from
        :py:attr:`input_parameters.ring.Ring.n_turns`
    ring_circumference : float
        Inherited from
        :py:attr:`input_parameters.ring.Ring.ring_circumference`
    section_length : float
        Length :math:`L_k` of the RF section; inherited from
        :py:attr:`input_parameters.ring.Ring.ring_length`
    length_ratio : float
        Fractional RF section length :math:`L_k/C`
    t_rev : float array [n_turns+1]
        Inherited from
        :py:attr:`input_parameters.ring.Ring.t_rev`
    momentum : float array [n_turns+1]
        Momentum program of the present RF section; inherited from
        :py:attr:`input_parameters.ring.Ring.momentum`
    beta : float array [n_turns+1]
        Relativistic beta of the present RF section; inherited from
        :py:attr:`input_parameters.ring.Ring.beta`
    gamma : float array [n_turns+1]
        Relativistic gamma of the present RF section; inherited from
        :py:attr:`input_parameters.ring.Ring.gamma`
    energy : float array [n_turns+1]
        Total energy of the present RF section; inherited from
        :py:attr:`input_parameters.ring.Ring.energy`
    delta_E : float array [n_turns]
        Time derivative of total energy of the present section; inherited from
        :py:attr:`input_parameters.ring.Ring.delta_E`
    alpha_order : int
        Inherited from
        :py:attr:`input_parameters.ring.Ring.alpha_order`
    charge : int
        Inherited from
        :py:attr:`beam.Particle.charge`
    eta_0 : float array [n_turns+1]
        Zeroth order slippage factor of the present section; inherited from
        :py:attr:`input_parameters.ring.Ring.eta_0`
    eta_1 : float array [n_turns+1]
        First order slippage factor of the present section; inherited from
        :py:attr:`input_parameters.ring.Ring.eta_1`
    eta_2 : float array [n_turns+1]
        Second order slippage factor of the present section; inherited from
        :py:attr:`input_parameters.ring.Ring.eta_2`
    sign_eta_0 : float array
        Sign of the eta_0 array
    harmonic : float matrix [n_rf, n_turns+1]
        Harmonic number for each rf system,
        :math:`h_{l,n}` [1]
    voltage : float matrix [n_rf, n_turns+1]
        Actual rf voltage of each harmonic system,
        :math:`V_{rf,l,n}` [V]
    empty : bool
        Flag to specify if the RFStation is empty
    phi_rf_d : float matrix [n_rf, n_turns+1]
        Designed rf cavity phase of each harmonic system,
        :math:`\phi_{d,l,n}` [rad]
    phi_rf : float matrix [n_rf, n_turns+1]
        Actual RF cavity phase of each harmonic system used for tracking,
        :math:`\phi_{rf,l,n}` [rad]. Initially the same as the designed phase.
    omega_rf_d : float matrix [n_rf, n_turns+1]
        Design RF angular frequency of the RF systems in the station
        :math:`\omega_{d,l,n} = \frac{h_{l,n} \beta_{l,n} c}{R_{s,n}}` [Hz]
    omega_rf : float matrix [n_rf, n_turns+1]
        Actual RF angular frequency of the RF systems in the station
        :math:`\omega_{rf,l,n} = \frac{h_{l,n} \beta_{l,n} c}{R_{s,n}}` [Hz].
        Initially the same as the designed angular frequency.
    phi_noise : None or float matrix [n_rf, n_turns+1]
        Programmed cavity phase noise for each RF harmonic.
    phi_modulation : None or float matrix [n_rf, n_turns+1]
        Programmed cavity phase modulation for each RF harmonic.
    dphi_rf : float matrix [n_rf]
        Accumulated RF phase error of each harmonic system
        :math:`\Delta \phi_{rf,l,n}` [rad]
    t_rf : float matrix [n_rf, n_turns+1]
        RF period :math:`\frac{2 \pi}{\omega_{rf,l,n}}` [s]
    phi_s : float array [n_turns+1]
        Synchronous phase for this section, calculated in
        :py:func:`input_parameters.rf_parameters.calculate_phi_s`
    Q_s : float array [n_turns+1]
        Synchrotron tune for this section, calculated in
        :py:func:`input_parameters.rf_parameters.calculate_Q_s`
    omega_s0 : float array [n_turns+1]
        Central synchronous angular frequency corresponding to Q_s (single
        harmonic, no intensity effects)
        :math:`\omega_{s,0} = Q_s \omega_{\text{rev}}` [1/s], where
        :math:`\omega_{\text{rev}}` is defined in
        :py:class:`input_parameters.ring.Ring`)
    RFStationOptions : RFStationOptions()
        The RFStationOptions is kept as an attribute of the RFStationg object
        for further usage.


    Examples
    --------
    >>> # To declare a double-harmonic RF system for protons:
    >>>
    >>> n_turns = 10
    >>> C = 26659
    >>> alpha_0 = 3.21e-4
    >>> momentum = 450e9
    >>> ring = Ring(C, alpha_0, momentum, n_turns)
    >>> rf_station = RFStation(ring, [35640, 71280], [6e6, 6e5], [0, 0], 2)

    """
    #TODO: move offsets to tracking only version
    def __init__(self, Ring, harmonic, voltage, phi_rf_d,
                 section_index=1, omega_rf_offset=None, phi_rf_offset=None):

        # Different indices
        self.counter = [int(0)]
        self.section_index = int(section_index - 1)
        
        
        if self.section_index < 0 \
                or self.section_index > Ring.n_sections - 1:
            raise RuntimeError("ERROR in RFStation: section_index out of" +
                               " allowed range!")


        #Coercion of voltage to RF_section_function datatype
        if not isinstance(voltage, rfProgs.voltage_program):
            if not hasattr(voltage, '__iter__'):
                voltage = (voltage, )
            if isinstance(voltage, dict):
                useV = []
                for h in harmonic:
                    useV.append(voltage.pop(h, 0))
                if len(voltage) != 0:
                    raise RuntimeError("Unrecognised harmonics in voltage")
                voltage = useV

            try:
                voltage = rfProgs.voltage_program(*voltage, 
                                                  harmonics = harmonic, 
                                                  interpolation = 'linear')
            except excpt.DataDefinitionError:
                voltage = rfProgs.voltage_program(*voltage, 
                                                  harmonics = harmonic)

        #Coercion of phase to RF_section_function datatype
        if not isinstance(phi_rf_d, rfProgs.phase_program):
            if not hasattr(phi_rf_d, '__iter__'):
                phi_rf_d = (phi_rf_d, )
            if isinstance(phi_rf_d, dict):
                usePhi = []
                for h in harmonic:
                    usePhi.append(phi_rf_d.pop(h, 0))
                if len(phi_rf_d) != 0:
                    raise RuntimeError("Unrecognised harmonics in phi_rf_d")
                phi_rf_d = usePhi
            try:
                phi_rf_d = rfProgs.phase_program(*phi_rf_d, 
                                                 harmonics = harmonic, 
                                                 interpolation = 'linear')
            except excpt.DataDefinitionError:
                phi_rf_d = rfProgs.phase_program(*phi_rf_d, 
                                                harmonics = harmonic)

        if not hasattr(harmonic, '__iter__'):
            harmonic = (harmonic,)

        assrt.equal_arrays(harmonic, voltage.harmonics, 
                           phi_rf_d.harmonics,
            msg = 'Declared harmonics and harmonics of voltage and phase'
            + ' functions do not all match', exception = excpt.InputDataError)
        
        self.n_rf = len(voltage.harmonics)
        

        self._ring_pars(Ring)

        # The order alpha_order used here can be replaced by Ring.alpha_order
        # when the assembler can differentiate the cases 'simple' and 'full'
        # for the drift
        alpha_order = 2#Ring.alpha_order
        for i in range(alpha_order+1):
            try:
                dummy = getattr(Ring, 'eta_' + str(i))
            except AttributeError:
                setattr(self, "eta_%s" % i, 0)
            else:
                setattr(self, "eta_%s" % i, dummy[self.section_index])
        self.sign_eta_0 = np.sign(self.eta_0)

        # Reshape design voltage
        self.voltage = voltage.reshape(use_time = Ring.cycle_time, 
                                       use_turns = Ring.use_turns)

        self.harmonic = np.zeros(self.voltage.shape)
        for i, h in enumerate(harmonic):
            self.harmonic[i] = h
            
        # Checking if the RFStation is empty
        if np.sum(self.voltage) == 0:
            self.empty = True
        else:
            self.empty = False


        # Reshape design phase
        self.phi_rf_d = phi_rf_d.reshape(use_time = Ring.cycle_time, 
                                         use_turns = Ring.use_turns)


        # Calculating design rf angular frequency
        self.omega_rf_d = 2.*np.pi*self.beta*c*self.harmonic / \
                            (self.ring_circumference)
        
        
        # Calculating omega and phi offsets
        if omega_rf_offset is None:
            useoff = (0,)*self.harmonic.shape[0]
            omega_rf_offset = rfProgs.omega_offset(*useoff, 
                                                  harmonics=harmonic)

        if not isinstance(omega_rf_offset, rfProgs.omega_offset):

            if isinstance(omega_rf_offset, dict):
                useoff = []
                for h in harmonic:
                    useoff.append(omega_rf_offset.pop(h, 0))
                if len(omega_rf_offset) != 0:
                    raise RuntimeError("Unrecognised harmonics in phi_rf_d")
                omega_rf_offset = useoff

            try:
                omega_rf_offset = rfProgs.omega_offset(*omega_rf_offset, 
                                                 harmonics = harmonic, 
                                                 interpolation = 'linear')

            except excpt.DataDefinitionError:
                omega_rf_offset = rfProgs.omega_offset(*omega_rf_offset, 
                                                      harmonics = harmonic)

        self.omega_rf_offset = omega_rf_offset.reshape(self.harmonic[:,0],
                                                       Ring.cycle_time, 
                                                       Ring.use_turns)

        if phi_rf_offset is None:
            useoff = (0,)*self.harmonic.shape[0]
            phi_rf_offset = rfProgs.phase_offset(*useoff, 
                                                harmonics=harmonic)
        if not isinstance(phi_rf_offset, rfProgs.phase_offset):

            if isinstance(phi_rf_offset, dict):
                useoff = []
                for h in harmonic:
                    useoff.append(phi_rf_offset.pop(h, 0))
                if len(omega_rf_offset) != 0:
                    raise RuntimeError("Unrecognised harmonics in phi_rf_d")
                phi_rf_offset = useoff

            try:
                phi_rf_offset = rfProgs.phase_offset(*phi_rf_offset, 
                                                     harmonics = harmonic, 
                                                     interpolation = 'linear')

            except excpt.DataDefinitionError:
                phi_rf_offset = rfProgs.phase_offset(*phi_rf_offset, 
                                                   harmonics = harmonic)

        self.phi_rf_offset = phi_rf_offset.reshape(self.harmonic[:,0],
                                                   Ring.cycle_time, 
                                                   Ring.use_turns)
        
        deltaPhaseFromOmega = self.omega_rf_offset.calc_delta_phase(
                                                    Ring.omega_rev)
        deltaOmegaFromPhase = self.phi_rf_offset.calc_delta_omega(
                                                    Ring.omega_rev)        
        
        self.phi_rf = np.array(self.phi_rf_d)
        self.omega_rf = np.array(self.omega_rf_d)
        
        self.phi_rf += deltaPhaseFromOmega + self.phi_rf_offset
        self.omega_rf += deltaOmegaFromPhase + self.omega_rf_offset
        
        # Copy of the desing rf programs in the one used for tracking
        # and that can be changed by feedbacks
        self.dphi_rf = np.zeros(self.n_rf)
        self.t_rf = 2*np.pi / self.omega_rf

        # From helper functions
        self.phi_s = calculate_phi_s(self, self.Particle)
        self.Q_s = calculate_Q_s(self, self.Particle)
        self.omega_s0 = self.Q_s*Ring.omega_rev


    @classmethod
    def from_rf_systems(cls, Ring, *args, section_index=1):

        self = object.__new__(cls)
        self.section_index = int(section_index-1)
        self._ring_pars(Ring)

        rfShape = [len(args), len(self.cycle_time)]

        self.voltage = rfProgs.voltage_program.zeros(rfShape)
        self.phi_rf_d = rfProgs.phase_program.zeros(rfShape)
        self.harmonic = np.zeros(rfShape)
        
        for i, a in enumerate(args):
            self.voltage[i], self.phi_rf_d[i], self.harmonic[i] \
                            = a.sample(self.cycle_time, self.use_turns)

        return self


    def _ring_pars(self, Ring):

        self.Particle = Ring.Particle
        self.n_turns = Ring.n_turns
        self.cycle_time = Ring.cycle_time
        self.ring_circumference = Ring.circumference
        self.section_length = Ring.section_length[self.section_index]
        self.length_ratio = self.section_length/self.ring_circumference
        self.t_rev = Ring.t_rev
        self.momentum = Ring.momentum[self.section_index]
        self.beta = Ring.beta[self.section_index]
        self.gamma = Ring.gamma[self.section_index]
        self.energy = Ring.energy[self.section_index]
        self.delta_E = Ring.delta_E[self.section_index]
        self.alpha_orders = Ring.alpha_orders
        self.charge = self.Particle.charge
        self.use_turns = Ring.use_turns.astype(int)


    def eta_tracking(self, beam, counter, dE):
        r"""Function to calculate the slippage factor as a function of the
        energy offset :math:`\Delta E` of the particle. The slippage factor
        of the :math:`i` th order is :math:`\eta(\delta) = \sum_{i}(\eta_i \,
        \delta^i) = \sum_{i} \left(\eta_i \, \left[ \frac{\Delta E}
        {\beta_s^2 E_s} \right]^i \right)`

        """

        if self.alpha_order == 0:
            return self.eta_0[counter]
        else:
            eta = 0
            delta = dE/(beam.beta**2 * beam.energy)
            for i in range(self.alpha_order+1):
                eta_i = getattr(self, 'eta_' + str(i))[counter]
                eta += eta_i * (delta**i)
            return eta


    def parameters_at_time(self, cycle_moments):
        """ Function to return various RF parameters at a specific moment in
        time.

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
        voltage = []
        phase = []
        harmonic = []
        omega = []
        for v, p, h, o in zip(self.voltage, self.phi_rf_d, self.harmonic,
                              self.omega_rf_d):
            voltage.append(np.interp(cycle_moments, self.cycle_time, v))
            phase.append(np.interp(cycle_moments, self.cycle_time, p))
            harmonic.append(np.interp(cycle_moments, self.cycle_time, h))
            omega.append(np.interp(cycle_moments, self.cycle_time, o))
        parameters = {}
        parameters['voltage'] = voltage
        parameters['phi_rf_d'] = phase
        parameters['harmonic'] = harmonic
        parameters['omega_rf_d'] = omega

        return parameters
    
    
    def parameters_at_turn(self, turn):

        try:
            sample = np.where(self.use_turns == turn)[0][0]
        except IndexError:
            raise excpt.InputError("turn " + str(turn) + " has not been "
                                   + "stored for the specified interpolation")
        else:
            return self.parameters_at_sample(sample)


    def parameters_at_sample(self, sample):
        
        parameters = {}
        parameters['voltage'] = self.voltage[:, sample]
        parameters['phi_rf_d'] = self.phi_rf_d[:, sample]
        parameters['harmonic'] = self.harmonic[:, sample]
        parameters['omega_rf_d'] = self.omega_rf_d[:, sample]
        
        return parameters
        

def calculate_Q_s(RFStation, Particle=Proton()):
    r""" Function calculating the turn-by-turn synchrotron tune for
    single-harmonic RF, without intensity effects.

    Parameters
    ----------
    RFStation : class
        An RFStation type class.
    Particle : class
        A Particle type class; default is Proton().

    Returns
    -------
    float
        Synchrotron tune.

    """

    return np.sqrt(RFStation.harmonic[0]*np.abs(Particle.charge) *
                   RFStation.voltage[0] *
                   np.abs(RFStation.eta_0*np.cos(RFStation.phi_s)) /
                   (2*np.pi*RFStation.beta**2*RFStation.energy))


def calculate_phi_s(RFStation, Particle=Proton(),
                    accelerating_systems='as_single'):
    r"""Function calculating the turn-by-turn synchronous phase according to
    the parameters in the RFStation object. The phase is expressed in
    the lowest RF harmonic and with respect to the RF bucket (see the equations
    of motion defined for BLonD). The returned value is given in the range [0,
    2*Pi]. Below transition, the RF wave is shifted by Pi w.r.t. the time
    reference.

    The accelerating_systems option can be set to

    * 'as_single' (default): the synchronous phase is calculated analytically
      taking into account the phase program (RFStation.phi_offset).
    * 'all': the synchronous phase is calculated numerically by finding the
      minimum of the potential well; no intensity effects included. In case of
      several minima, the deepest is taken. **WARNING:** in case of RF
      harmonics with comparable voltages, this may lead to inconsistent
      values of phi_s.
    * 'first': not yet implemented. Its purpose should be to adjust the
      RFStation.phi_offset of the higher harmonics so that only the
      main harmonic is accelerating.

    Parameters
    ----------
    RFStation : class
        An RFStation type class.
    Particle : class
        A Particle type class; default is Proton().
    accelerating_systems : str
        Choice of accelerating systems; or options, see list above.

    Returns
    -------
    float
        Synchronous phase.

    """

    eta0 = RFStation.eta_0

    if accelerating_systems == 'as_single':

        if RFStation.delta_E.shape[0] == RFStation.momentum.shape[0]:
            denergy = RFStation.delta_E.copy()
        else:
            denergy = np.append(RFStation.delta_E, RFStation.delta_E[-1])
        acceleration_ratio = denergy/(Particle.charge*RFStation.voltage[0, :])
        acceleration_test = np.where((acceleration_ratio > -1) *
                                     (acceleration_ratio < 1) is False)[0]

        # Validity check on acceleration_ratio
        if acceleration_test.size > 0:
            print("WARNING in calculate_phi_s(): acceleration is not " +
                  "possible (momentum increment is too big or voltage too " +
                  "low) at index " + str(acceleration_test))

        phi_s = np.arcsin(acceleration_ratio)

        # Identify where eta swaps sign
        eta0_middle_points = (eta0[1:] + eta0[:-1])/2
        eta0_middle_points = np.append(eta0_middle_points, eta0[-1])
        index = np.where(eta0_middle_points > 0)[0]
        index_below = np.where(eta0_middle_points < 0)[0]

        # Project phi_s in correct range
        phi_s[index] = (np.pi - phi_s[index]) % (2*np.pi)
        phi_s[index_below] = (np.pi + phi_s[index_below]) % (2*np.pi)

        return phi_s

    elif accelerating_systems == 'all':

        phi_s = np.zeros(len(RFStation.voltage[0, 1:]))

        for indexTurn in range(len(RFStation.delta_E)):

            totalRF = 0
            if np.sign(eta0[indexTurn]) > 0:
                phase_array = np.linspace(
                    -float(RFStation.phi_rf[0, indexTurn+1]),
                    -float(RFStation.phi_rf[0, indexTurn+1]) + 2*np.pi, 1000)
            else:
                phase_array = np.linspace(
                    -float(RFStation.phi_rf[0, indexTurn+1]) - np.pi,
                    -float(RFStation.phi_rf[0, indexTurn+1]) + np.pi, 1000)

            for indexRF in range(len(RFStation.voltage[:, indexTurn+1])):
                totalRF += RFStation.voltage[indexRF, indexTurn+1] * \
                    np.sin(RFStation.harmonic[indexRF, indexTurn+1] /
                           np.min(RFStation.harmonic[:, indexTurn+1]) *
                           phase_array +
                           RFStation.phi_rf[indexRF, indexTurn+1])

            potential_well = - cumtrapz(
                np.sign(eta0[indexTurn])*(totalRF -
                                          RFStation.delta_E[indexTurn] /
                                          abs(Particle.charge)),
                dx=phase_array[1]-phase_array[0], initial=0)

            phi_s[indexTurn] = np.mean(phase_array[
                potential_well == np.min(potential_well)])

        phi_s = np.insert(phi_s, 0, phi_s[0]) + RFStation.phi_rf[0, :]
        phi_s[eta0 < 0] += np.pi
        phi_s = phi_s % (2*np.pi)

        return phi_s

    elif accelerating_systems == 'first':

        print("WARNING in calculate_phi_s(): accelerating_systems 'first'" +
              " not yet implemented")
        pass
    else:
        raise RuntimeError("ERROR in calculate_phi_s(): unrecognised" +
                           " accelerating_systems option")
