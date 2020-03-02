# coding: utf8
# Copyright 2014-2019 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Module gathering all general input parameters used for the simulation.**
    :Authors: **Alexandre Lasheen**, **Danilo Quartullo**, **Helga Timko**
'''

from __future__ import division
from builtins import str, range, object
import numpy as np
import warnings
from scipy.constants import c
import sys

from ...devtools import exceptions as excpt
from ...devtools import assertions as assrt
from ..beam import beam
from ...datatypes import datatypes as dTypes
from ...utilities import timing as tmng
from ...utilities import rel_transforms as rt


class Ring:
    r""" Class containing the general properties of the synchrotron that are
    independent of the RF system or the beam.

    The index :math:`n` denotes time steps, :math:`k` ring segments/sections
    and :math:`i` momentum compaction orders.

    Parameters
    ----------
    ring_length : float (opt: float array [n_sections])
        Length [m] of the n_sections ring segments of the synchrotron.
        An RF station, a synchrotron radiation kick, and/or an impedance kick
        can be included at the end of each ring section.
    alpha_0 : float (opt: float array/matrix [n_sections, n_turns+1])
        Momentum compaction factor of zeroth order :math:`\alpha_{0,k,i}` [1];
        can be input as single float or as a program of (n_turns + 1) turns
        (should be of the same size as synchronous_data).
        In case of higher order momentum compaction, check the
        documentation for the inputs: alpha_order, alpha_1, alpha_2
    synchronous_data : float (opt: float array/matrix [n_sections, n_turns+1])
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
    Particle : class
        A Particle-based class defining the primary, synchronous particle (mass
        and charge) that is reference for the momentum/energy in the ring.
    n_turns : int
        Optional: Number of turns :math:`n` [1] to be simulated.
        If a synchronous_data program is passed as a tuple (see below),
        the number of turns will be overwritten depending on the length in time
        of the program
    synchronous_data_type : str
        Optional: Choice of 'synchronous_data' type; can be 'momentum'
        (default), 'total energy', 'kinetic energy' or 'bending field'
        (requires bending_radius to be defined)
    bending_radius : float
        Optional: Radius [m] of the bending magnets,
        required if 'bending field' is set for the synchronous_data_type
    n_sections : int
        Optional: number of ring sections/segments; default is 1
    alpha_1 : float (opt: float array/matrix [n_sections, n_turns+1])
        Momentum compaction factor of first order
        :math:`\alpha_{1,k,i}` [1]; can be input as single float or as a
        program of (n_turns + 1) turns (should be of the same size as
        synchronous_data and alpha_0).
    alpha_2 : float (opt: float array/matrix [n_sections, n_turns+1])
        Optional : Momentum compaction factor of second order
        :math:`\alpha_{2,k,i}` [1]; can be input as single float or as a
        program of (n_turns + 1) turns (should be of the same size as
        synchronous_data and alpha_0).
    RingOptions : class
        Optional : A RingOptions-based class with default options to check the
        input and initialize the momentum program for the simulation.
        This object defines the interpolation scheme, plotting options, etc.
        The options for this object can be adjusted and passed to the Ring
        object.

    Attributes
    ----------
    ring_circumference : float
        Circumference of the synchrotron. Sum of ring segment lengths,
        :math:`C = \sum_k L_k` [m]
    ring_radius : float
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

    def __init__(self, ring_length, alpha, synchronous_data, Particle,
                 bending_radius=None, **kwargs):

        # Ring length and checks
        self.ring_length = np.array(ring_length, ndmin=1, dtype=float)
        self.ring_circumference = np.sum(self.ring_length)
        self.ring_radius = self.ring_circumference/(2*np.pi)

        if bending_radius is not None:
            self.bending_radius = float(bending_radius)
        else:
            self.bending_radius = bending_radius

        # Primary particle mass and charge used for energy calculations
        if isinstance(Particle, beam.Particle):
            self.Particle = Particle
        else:
            self.Particle = beam.make_particle(Particle)
            
        # Reshaping the input synchronous data to the adequate format and
        # get back the momentum program from RingOptions
        if not isinstance(synchronous_data, dTypes.ring_program):
                synchronous_data = dTypes.ring_program(synchronous_data)

        if synchronous_data.shape[0] != len(self.ring_length):
            raise excpt.InputDataError("ERROR in Ring: Number of sections "
                                       +"and ring length size do not match!")

        t_start = kwargs.pop('t_start', 0)
        t_stop = kwargs.pop('t_start', np.inf)
        interp_time = kwargs.pop('interp_time', 0)
        
        if not hasattr(interp_time, '__iter__'):
            interp_time = (interp_time, )
        
        sample_func, start, stop = tmng.time_from_sampling(*interp_time)
        
        if t_start > start:
            start = t_start
        if t_stop < stop:
            stop = t_stop
        
        synchronous_data.convert(self.Particle.mass, self.Particle.charge, 
                                 self.bending_radius)
        
        self.momentum = synchronous_data.preprocess(self.Particle.mass, 
                                    self.ring_circumference, sample_func, 
                                    'linear', start, stop)

        self.n_sections = self.momentum.shape[0]-2
        self.cycle_time = self.momentum[1]
        self.use_turns = self.momentum[0]
        # Updating the number of turns in case it was changed after ramp
        # interpolation
        self.n_turns = self.momentum.n_turns

        # Derived from momentum
        self.beta = rt.mom_to_beta(self.momentum[2:], self.Particle.mass)
        self.gamma = rt.mom_to_gamma(self.momentum[2:], self.Particle.mass)
        self.energy = rt.mom_to_energy(self.momentum[2:], self.Particle.mass)
        self.kin_energy = rt.mom_to_kin_energy(self.momentum[2:], 
                                               self.Particle.mass)
        self.t_rev = np.dot(self.ring_length, 1/(self.beta*c))            
        self.delta_E = np.diff(self.energy, axis=1)
        if self.n_turns > len(self.use_turns):
            self._recalc_delta_E()
        
        self.momentum = self.momentum[2:]
            
        self.f_rev = 1/self.t_rev
        self.omega_rev = 2*np.pi*self.f_rev

        # Momentum compaction, checks, and derived slippage factors
        
        if not hasattr(alpha, '__iter__'):
            alpha = (alpha, )
            
        if isinstance(alpha, dict):
            try:
                if not all([k%1 == 0 for k in alpha.keys()]):
                    raise TypeError
            except TypeError:
                raise excpt.InputError("If alpha is dict all keys must be "
                                       + "numeric and integer")

            maxAlpha = np.max(tuple(alpha.keys())).astype(int)
            alpha = [alpha.pop(i, 0) for i in range(maxAlpha+1)]

        for i, a in enumerate(alpha):
            if not isinstance(a, dTypes.momentum_compaction):
                a = dTypes.momentum_compaction(a, order = i)
            setattr(self, 'alpha_'+str(i), a.reshape(self.n_sections, 
                                                    self.cycle_time))
            setattr(self, 'eta_'+str(i), np.zeros([self.n_sections, 
                                                    len(self.use_turns)]))
        self.alpha_order = i
        
        for i in range(3 - self.alpha_order):
            setattr(self, 'alpha_'+str(i), np.zeros([self.n_sections, 
                                                    len(self.use_turns)]))
            setattr(self, 'eta_'+str(i), np.zeros([self.n_sections, 
                                                    len(self.use_turns)]))
            

        # Slippage factor derived from alpha, beta, gamma
        self.eta_generation()


    def eta_generation(self):
        """ Function to generate the slippage factors (zeroth, first, and
        second orders, see [1]_) from the momentum compaction and the
        relativistic beta and gamma program through the cycle.

        References
        ----------
        .. [1] "Accelerator Physics," S. Y. Lee, World Scientific,
                Third Edition, 2012.
        """

        for i in range(self.alpha_order+1):
            getattr(self, '_eta' + str(i))()


    def _eta0(self):
        """ Function to calculate the zeroth order slippage factor eta_0 """

        for i in range(0, self.n_sections):
            self.eta_0[i] = self.alpha_0[i] - self.gamma[i]**(-2.)

    def _eta1(self):
        """ Function to calculate the first order slippage factor eta_1 """

        for i in range(0, self.n_sections):
            self.eta_1[i] = 3*self.beta[i]**2/(2*self.gamma[i]**2) + \
                self.alpha_1[i] - self.alpha_0[i]*self.eta_0[i]

    def _eta2(self):
        """ Function to calculate the second order slippage factor eta_2 """

        for i in range(0, self.n_sections):
            self.eta_2[i] = - self.beta[i]**2*(5*self.beta[i]**2 - 1) / \
                (2*self.gamma[i]**2) + self.alpha_2[i] - 2*self.alpha_0[i] *\
                self.alpha_1[i] + self.alpha_1[i] / self.gamma[i]**2 + \
                self.alpha_0[i]**2*self.eta_0[i] - 3*self.beta[i]**2 * \
                self.alpha_0[i]/(2*self.gamma[i]**2)

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
        parameters['delta_E'] = np.interp(cycle_moments,
                                          self.cycle_time[1:],
                                          self.delta_E[0])
        parameters['charge'] = self.Particle.charge

        return parameters

    #TODO: fix len(delta_E) for non 't_rev' interpolation
    def _recalc_delta_E(self):
        """
        Function to recalculate delta_E.
        If interpolation is not done on a turn-by-turn basis the delta_E will
        not be correct.  This function recalculates it to give the correct 
        value for each turn.
        """
        
        for section in range(self.n_sections):
            ENow = self.energy[section]
            ENext = np.interp(self.cycle_time + self.t_rev, self.cycle_time, 
                              ENow)
            
            self.delta_E[section][:] = (ENext - ENow)[:-1]

