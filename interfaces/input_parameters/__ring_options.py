# coding: utf8
# Copyright 2014-2019 CERN. This software is distributed under the
# terms of the GNU General Public License version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.md.
# In applying this license, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Function(s) for pre-processing input data**

:Authors: **Helga Timko**, **Alexandre Lasheen**, **Danilo Quartullo**,
    **Simon Albright**
'''

# General imports old
from __future__ import division
from builtins import str, range
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
from scipy.interpolate import splrep, splev
from ...devtools.path import makedir
import warnings
import sys
import numbers

# General imports
import scipy.constants as cont

# BLonD_Common imports
from ...devtools import exceptions as excpt
from ...datatypes import datatypes as dTypes
from ...utilities import timing


'''
WARNING: This module is deprecated and kept only for reference before deletion!
'''


class _RingOptions(object):
    r""" Class to preprocess the synchronous data for Ring, interpolating it to
    every turn.

    Parameters
    ----------
    interpolation : str
        Interpolation options for the data points. Available options are
        'linear' (default), 'cubic', and 'derivative'
    smoothing : float
        Smoothing value for 'cubic' interpolation
    flat_bottom : int
        Number of turns to be added on flat bottom; default is 0. Constant
        extrapolation is used for the synchronous data
    flat_top : int
        Number of turns to be added on flat top; default is 0. Constant
        extrapolation is used for the synchronous data
    t_start : float or None
        Starting time from which the time array input should be taken into
        account; default is None
    t_end : float or None
        Last time up to which the time array input should be taken into
        account; default is None
    interp_time : 't_rev', float or array
        Time on which the momentum program should be interpolated;
        default is 't_rev' to compute at each revolution period (for tracking)
        if a float is passed, will interpolate at regular time
        if an array/list is passed, will interpolate at given time
    plot : bool
        Option to plot interpolated arrays; default is False
    figdir : str
        Directory to save optional plot; default is 'fig'
    figname : str
        Figure name to save optional plot; default is 'preprocess_ramp'
    sampling : int
        Decimation value for plotting; default is 1

    """
    def __init__(self, interpolation='linear', smoothing=0, flat_bottom=0,
                 flat_top=0, t_start=None, t_end=None, interp_time='t_rev',
                 plot=False, figdir='fig', figname='preprocess_ramp',
                 sampling=1):

        if interpolation in ['linear', 'cubic', 'derivative']:
            self.interpolation = str(interpolation)
        else:
            raise excpt.InputError("ERROR: Interpolation scheme in " +
                               "PreprocessRamp not recognised. Aborting...")

        self.smoothing = float(smoothing)

        if flat_bottom < 0:
            raise excpt.MomentumError("ERROR: flat_bottom value in " 
                                      + "PreprocessRamp not recognised. "
                                      + "Aborting...")
        else:
            self.flat_bottom = int(flat_bottom)

        if flat_top < 0:
            raise excpt.MomentumError("ERROR: flat_top value in PreprocessRamp"
                                      + " not recognised. Aborting...")
        else:
            self.flat_top = int(flat_top)

        self.t_start = t_start
        self.t_end = t_end

        if (interp_time != 't_rev') and \
                not isinstance(interp_time, float) and \
                not isinstance(interp_time, np.ndarray) and \
                not isinstance(interp_time, list):
            raise RuntimeError("ERROR: interp_time value in PreprocessRamp" +
                               " not recognised. Aborting...")
        else:
            self.interp_time = interp_time

        if (plot is True) or (plot is False):
            self.plot = bool(plot)
        else:
            raise TypeError("ERROR: plot value in PreprocessRamp" +
                               " not recognised. Aborting...")

        self.figdir = str(figdir)
        self.figname = str(figname)
        if sampling > 0:
            self.sampling = int(sampling)
        else:
            raise TypeError("ERROR: sampling value in PreprocessRamp" +
                               " not recognised. Aborting...")

    def reshape_data(self, input_data, n_turns, n_sections,
                      input_to_momentum=False,
                     synchronous_data_type='momentum', mass=None, charge=None,
                     circumference=None, bending_radius=None):
        r"""Checks whether the user input is consistent with the expectation
        for the Ring object. The possibilites are detailed in the documentation
        of the Ring object.


        Parameters
        ----------
        input_data : Ring.synchronous_data, Ring.alpha_0,1,2
            Main input data to reshape
        n_turns : Ring.n_turns
            Number of turns the simulation should be. Note that if
            the input_data is passed as a tuple it is expected that the
            input_data is a program. Hence, the number of turns may not
            correspond to the input one and will be overwritten
        n_sections : Ring.n_sections
            The number of sections of the ring. The simulation is stopped
            if the input_data shape does not correspond to the expected number
            of sections.
        interp_time : str or float or float array
            Optional : defines the time on which the program will be
            interpolated. If 't_rev' is passed and if the input_data is
            momentum (see input_to_momentum option) the momentum program
            is interpolated on the revolution period (see preprocess()
            function). If a float or a float array is passed, the program
            is interpolated on that input ; default is 't_rev'
        input_to_momentum : bool
            Optional : flags if the input_data is the momentum program, the
            options defined below become necessary for conversion
        synchronous_data_type : str
            Optional : to be passed to the convert_data function if
            input_to_momentum ; default is 'momentum'
        mass : Ring.Particle.mass
            Optional : the mass of the particles in [eV/c**2] ; default is None
        charge : Ring.Particle.charge
            Optional : the charge of the particles in units of [e] ;
            default is None
        circumference : Ring.circumference
            Optional : the circumference of the ring ; default is None
        bending_radius : Ring.bending_radis
            Optional : the bending radius of magnets ; default is None

        Returns
        -------
        output_data
            Returns the data with the adequate shape for the Ring object

        """

        #TEST LOOP FOR DATA TYPE OBJECT
        if hasattr(input_data, 'data_type'):
            data_type = input_data.data_type
            if data_type[0] != 'momentum':
                raise RuntimeError("Input data is not a momentum_program")
            
            #TO DO: Needs modifying for single valued multi-section functions
            if data_type[1] == 'single':
                if input_to_momentum:
                    input_data = convert_data(input_data, mass, charge,
                                              synchronous_data_type,
                                              bending_radius)
                return input_data * np.ones((n_sections, n_turns+1))

            if data_type[2] == 'single_section':
                input_data = (input_data, )

            output_data = []
            for sect in range(n_sections):
                if data_type[1] == 'by_turn':
                    inputValues = input_data[sect]
                elif data_type[1] == 'by_time':
                    inputValues = input_data[sect][1]
                else:
                    raise RuntimeError("Input data type not recognised, " \
                                       + "should be by_turn or by_time")
                
                if input_to_momentum:
                    inputValues = convert_data(inputValues, mass, charge, \
                                                  synchronous_data_type, \
                                                  bending_radius)
                
                if data_type[1] == 'by_turn':
                    output_data.append(inputValues)
                    continue
                
                inputTime = input_data[sect][0]
                
                if self.interp_time == 't_rev':
                    output_data.append(self.preprocess(
                            mass,
                            circumference,
                            inputTime, inputValues)[1])
                else:
                    try:
                        iter(self.interp_time)
                    except TypeError:
                        self.interp_time = np.arange(inputTime[0], inputTime[-1], \
                                                float(self.interp_time))
                
                    output_data.append(np.interp(self.interp_time, inputTime, \
                                                     inputValues))
                
            output_data = np.array(output_data, ndmin=2, dtype=float)
    
            return output_data
        #END TEST LOOP FOR DATA TYPE OBJECT                    

        # TO BE IMPLEMENTED: if you pass a filename the function reads the file
        # and reshape the data
        if isinstance(input_data, str):
            pass

        # If single float, expands the value to match the input number of turns
        # and sections
        if isinstance(input_data, float) or isinstance(input_data, int):
            input_data = float(input_data)
            if input_to_momentum:
                input_data = convert_data(input_data, mass, charge,
                                          synchronous_data_type,
                                          bending_radius)
            output_data = input_data * np.ones((n_sections, n_turns+1))

        # If tuple, separate time and synchronous data and check data
        elif isinstance(input_data, tuple):

            output_data = []

            # If there is only one section, it is expected that the user passes
            # a tuple with (time, data). However, the user can also pass a
            # tuple which size is the number of section as ((time, data), ).
            # and this if condition takes this into account
            if (n_sections == 1) and (len(input_data) > 1):
                input_data = (input_data, )

            if len(input_data) != n_sections:
                #InputDataError
                raise RuntimeError("ERROR in Ring: the input data " +
                                   "does not match the number of sections")

            # Loops over all the sections to interpolate the programs, appends
            # the results on the output_data list which is afterwards
            # converted to a numpy.array
            for index_section in range(n_sections):
                input_data_time = input_data[index_section][0]
                input_data_values = input_data[index_section][1]

                if input_to_momentum:
                    input_data_values = convert_data(input_data_values, mass,
                                                     charge,
                                                     synchronous_data_type,
                                                     bending_radius)

                if len(input_data_time) \
                        != len(input_data_values):
                    #InputDataError
                    raise RuntimeError("ERROR in Ring: synchronous data " +
                                       "does not match the time data")

                if input_to_momentum and (self.interp_time == 't_rev'):
                    output_data.append(self.preprocess(
                        mass,
                        circumference,
                        input_data_time,
                        input_data_values)[1])

                elif isinstance(self.interp_time, float) or \
                        isinstance(self.interp_time, int):
                    self.interp_time = float(self.interp_time)
                    self.interp_time = np.arange(
                        input_data_time[0],
                        input_data_time[-1],
                        self.interp_time)

                    output_data.append(np.interp(
                        self.interp_time,
                        input_data_time,
                        input_data_values))

                elif isinstance(self.interp_time, np.ndarray):
                    output_data.append(np.interp(
                        self.interp_time,
                        input_data_time,
                        input_data_values))

            output_data = np.array(output_data, ndmin=2, dtype=float)

        # If array/list, compares with the input number of turns and
        # if synchronous_data is a single value converts it into a (n_turns+1)
        # array
        elif isinstance(input_data, np.ndarray) or \
                isinstance(input_data, list):

            input_data = np.array(input_data, ndmin=2, dtype=float)

            if input_to_momentum:
                input_data = convert_data(input_data, mass, charge,
                                          synchronous_data_type,
                                          bending_radius)

            output_data = np.zeros((n_sections, n_turns+1), dtype=float)

            # If the number of points is exactly the same as n_rf, this means
            # that the rf program for each harmonic is constant, reshaping
            # the array so that the size is [n_sections,1] for successful
            # reshaping
            if input_data.size == n_sections:
                input_data = input_data.reshape((n_sections, 1))

            if len(input_data) != n_sections:
                #InputDataError
                raise RuntimeError("ERROR in Ring: the input data " +
                                   "does not match the number of sections")

            for index_section in range(len(input_data)):
                if len(input_data[index_section]) == 1:
                    output_data[index_section] = input_data[index_section] * \
                                                np.ones(n_turns+1)

                elif len(input_data[index_section]) == (n_turns+1):
                    output_data[index_section] = np.array(
                        input_data[index_section])

                else:
                    #InputDataError
                    raise RuntimeError("ERROR in Ring: The input data " +
                                       "does not match the proper length " +
                                       "(n_turns+1)")

        return output_data

    def preprocess(self, mass, circumference, time, momentum):
        r"""Function to pre-process acceleration ramp data, interpolating it to
        every turn. Currently it works only if the number of RF sections is
        equal to one, to be extended for multiple RF sections.

        Parameters
        ----------
        mass : float
            Particle mass [eV]
        circumference : float
            Ring circumference [m]
        time : float array
            Time points [s] corresponding to momentum data
        momentum : float array
            Particle momentum [eV/c]

        Returns
        -------
        float array
            Cumulative time [s]
        float array
            Interpolated momentum [eV/c]

        """

        # Some checks on the options
        if ((self.t_start is not None) and (self.t_start < time[0])) or \
                ((self.t_end is not None) and (self.t_end > time[-1])):
                #InputDataError
                raise RuntimeError("ERROR: [t_start, t_end] should be " +
                                   "included in the passed time array.")

        # Obtain flat bottom data, extrapolate to constant
        beta_0 = np.sqrt(1/(1 + (mass/momentum[0])**2))
        T0 = circumference/(beta_0*c)  # Initial revolution period [s]
        shift = time[0] - self.flat_bottom*T0
        time_interp = shift + T0*np.arange(0, self.flat_bottom+1)
        beta_interp = beta_0*np.ones(self.flat_bottom+1)
        momentum_interp = momentum[0]*np.ones(self.flat_bottom+1)

        time_interp = time_interp.tolist()
        beta_interp = beta_interp.tolist()
        momentum_interp = momentum_interp.tolist()

        time_start_ramp = np.max(time[momentum == momentum[0]])
        time_end_ramp = np.min(time[momentum == momentum[-1]])

        # Interpolate data recursively
        if self.interpolation == 'linear':

            time_interp.append(time_interp[-1]
                               + circumference/(beta_interp[0]*c))

            i = self.flat_bottom
            for k in range(1, len(time)):

                while time_interp[i+1] <= time[k]:

                    momentum_interp.append(
                        momentum[k-1] + (momentum[k] - momentum[k-1]) *
                        (time_interp[i+1] - time[k-1]) /
                        (time[k] - time[k-1]))

                    beta_interp.append(
                        np.sqrt(1/(1 + (mass/momentum_interp[i+1])**2)))

                    time_interp.append(
                        time_interp[i+1] + circumference/(beta_interp[i+1]*c))

                    i += 1

        elif self.interpolation == 'cubic':

            interp_funtion_momentum = splrep(
                time[(time >= time_start_ramp) * (time <= time_end_ramp)],
                momentum[(time >= time_start_ramp) * (time <= time_end_ramp)],
                s=self.smoothing)

            i = self.flat_bottom

            time_interp.append(
                time_interp[-1] + circumference / (beta_interp[0]*c))

            while time_interp[i] <= time[-1]:

                if (time_interp[i+1] < time_start_ramp):

                    momentum_interp.append(momentum[0])

                    beta_interp.append(
                        np.sqrt(1/(1 + (mass/momentum_interp[i+1])**2)))

                    time_interp.append(
                        time_interp[i+1] + circumference/(beta_interp[i+1]*c))

                elif (time_interp[i+1] > time_end_ramp):

                    momentum_interp.append(momentum[-1])

                    beta_interp.append(
                        np.sqrt(1/(1 + (mass/momentum_interp[i+1])**2)))

                    time_interp.append(
                        time_interp[i+1] + circumference/(beta_interp[i+1]*c))

                else:

                    momentum_interp.append(
                        splev(time_interp[i+1], interp_funtion_momentum))

                    beta_interp.append(
                        np.sqrt(1/(1 + (mass/momentum_interp[i+1])**2)))

                    time_interp.append(
                        time_interp[i+1] + circumference/(beta_interp[i+1]*c))

                i += 1

        # Interpolate momentum in 1st derivative to maintain smooth B-dot
        elif self.interpolation == 'derivative':

            momentum_initial = momentum_interp[0]
            momentum_derivative = np.gradient(momentum)/np.gradient(time)

            momentum_derivative_interp = [0]*self.flat_bottom + \
                [momentum_derivative[0]]
            integral_point = momentum_initial

            i = self.flat_bottom

            time_interp.append(
                time_interp[-1] + circumference/(beta_interp[0]*c))

            while time_interp[i] <= time[-1]:

                derivative_point = np.interp(time_interp[i+1], time,
                                             momentum_derivative)
                momentum_derivative_interp.append(derivative_point)
                integral_point += (time_interp[i+1] - time_interp[i]) \
                    * derivative_point

                momentum_interp.append(integral_point)
                beta_interp.append(
                    np.sqrt(1/(1 + (mass/momentum_interp[i+1])**2)))

                time_interp.append(
                    time_interp[i+1] + circumference/(beta_interp[i+1]*c))

                i += 1

            # Adjust result to get flat top energy correct as derivation and
            # integration leads to ~10^-8 error in flat top momentum
            momentum_interp = np.asarray(momentum_interp)
            momentum_interp -= momentum_interp[0]
            momentum_interp /= momentum_interp[-1]
            momentum_interp *= momentum[-1] - momentum[0]

            momentum_interp += momentum[0]

        time_interp.pop()
        time_interp = np.asarray(time_interp)
        beta_interp = np.asarray(beta_interp)
        momentum_interp = np.asarray(momentum_interp)

        # Obtain flat top data, extrapolate to constant
        if self.flat_top > 0:
            time_interp = np.append(
                time_interp,
                time_interp[-1] + circumference*np.arange(1, self.flat_top+1)
                / (beta_interp[-1]*c))

            beta_interp = np.append(
                beta_interp, beta_interp[-1]*np.ones(self.flat_top))

            momentum_interp = np.append(
                momentum_interp,
                momentum_interp[-1]*np.ones(self.flat_top))

        # Cutting the input momentum on the desired cycle time
        if self.t_start is not None:
            initial_index = np.min(np.where(time_interp >= self.t_start)[0])
        else:
            initial_index = 0
        if self.t_end is not None:
            final_index = np.max(np.where(time_interp <= self.t_end)[0])+1
        else:
            final_index = len(time_interp)
        time_interp = time_interp[initial_index:final_index]
        momentum_interp = momentum_interp[initial_index:final_index]

        if self.plot:
            # Directory where longitudinal_plots will be stored
            makedir(self.figdir)

            # Plot
            plt.figure(1, figsize=(8, 6))
            ax = plt.axes([0.15, 0.1, 0.8, 0.8])
            ax.plot(time_interp[::self.sampling],
                    momentum_interp[::self.sampling],
                    label='Interpolated momentum')
            ax.plot(time, momentum, '.', label='input momentum', color='r',
                    markersize=0.5)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("p [eV]")
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax.legend = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                                   ncol=2, mode="expand", borderaxespad=0.)

            # Save figure
            fign = self.figdir + '/preprocess_' + self.figname
            plt.savefig(fign)
            plt.clf()

        return time_interp, momentum_interp



class RingOptions:
    r""" Class to preprocess the synchronous data for Ring, interpolating it to
    every turn.
    """

    def __init__(self, interp_time = 't_rev', interpolation = 'linear',
                 t_start = None, t_end = None, flat_bottom = 0, flat_top = 0, 
                 n_turns = np.inf, n_sections = 1):
        
        try:
            interp_time = interp_time.casefold()
        except AttributeError:
            pass
    
        if interp_time == 't_rev':
            interp_time = 0

        if isinstance(interp_time, numbers.Number) \
        or isinstance(interp_time, tuple) \
        or isinstance(interp_time, list):
            self.interp_time = interp_time            
        else:
            raise excpt.InputError("interp_time not recognised")
        
        if interpolation in ('linear', 'cubic', 'derivative'):
            self.interpolation = interpolation
        else:
            raise excpt.InputError("interpolation type not recognised")

        self.t_start = t_start
        self.t_end = t_end

        if flat_bottom >= 0:
            self.flat_bottom = flat_bottom
        else:
            raise excpt.InputError("flat_bottom must be 0 or greater")

        if flat_top >= 0:
            self.flat_top = flat_top
        else:
            raise excpt.InputError("flat_top must be 0 or greater")

        if n_turns is not np.inf:
            try:
                self.n_turns = int(n_turns)
                if n_turns % 1 != 0:
                    warnings.warn("non integer n_turns cast to int")
            except (ValueError, TypeError):
                if n_turns is None:
                    self.n_turns = np.inf
                else:
                    raise excpt.InputError("n_turns must be castable as int")
        else:
            self.n_turns = n_turns
        
        try:
            self.n_sections = int(n_sections)
            if n_sections % 1 != 0:
                warnings.warn("non integer n_sections cast to int")
        except (ValueError, TypeError):
            raise excpt.InputError("n_sections must be castable as int")
    
    
    def reshape_data(self, input_data, mass=None, charge=None,
                     circumference=None, bending_radius=None,
                     machine_times=None):
        
        if not isinstance(input_data, dTypes._function):
            raise excpt.InputError("data_type object expected")
            
        if isinstance(input_data, dTypes.ring_program) and \
            any(val is None for val in (mass, charge, circumference)):
            raise excpt.InputError("For interpolation of momentum type "
                                   + "functions mass, charge, circumference "
                                   + "must all be passed")
            

        data_type = input_data.data_type
        
        #TODO:  Mulit section
        if data_type[1] == 'single':
            if isinstance(input_data, dTypes.ring_program):
                input_data = convert_data(input_data, mass, charge,
                                          data_type[0], bending_radius)
            return input_data * np.ones((self.n_sections, len(self.use_turns)))
        
        if data_type[2] == 'single_section':
            input_data = (input_data, )
        
        output_data = []
        output_time = []
        for sect in range(self.n_sections):
            if data_type[1] == 'by_turn':
                inputValues = input_data[sect]
            elif data_type[1] == 'by_time':
                inputValues = input_data[sect][1]
            else:
                raise RuntimeError("Input data type not recognised, " 
                                   + "should be by_turn or by_time")
            
            if isinstance(input_data, dTypes.ring_program):
                inputValues = convert_data(inputValues, mass, charge, 
                                              data_type[0], bending_radius)
            
            if data_type[1] == 'by_turn':
                output_data.append(inputValues)
                continue
            
            inputTime = input_data[sect][0]
            if isinstance(input_data[sect], dTypes.ring_program):
                processTime, processData = self.preprocess(mass, circumference,
                                                           inputTime, 
                                                           inputValues)
                output_time.append(processTime)
                output_data.append(processData)
                
        output_time = np.array(output_time, ndmin=2, dtype=float)
        output_data = np.array(output_data, ndmin=2, dtype=float)

        return output_time, output_data
                

    def preprocess(self, mass, circumference, time, momentum, 
                   interpolation = None):

        if interpolation is None:
            interpolation = self.interpolation
        
        # Some checks on the options
        if ((self.t_start is not None) and (self.t_start < time[0])) or \
            ((self.t_end is not None) and (self.t_end > time[-1])):
            raise excpt.InputError("ERROR: [t_start, t_end] should be " +
                                   "included in the passed time array.")

        if self.t_start == None:
            self.t_start = time[0]
        if self.t_end == None:
            self.t_end = time[-1]

        # Obtain flat bottom data, extrapolate to constant
        
        time_func, start, end = timing.time_from_sampling(self.interp_time)

        if start > self.t_start:
            self.t_start = start
            warnings.warn("self.t_start being overwritten")

        if end < self.t_end:
            self.t_end = end
            warnings.warn("self.t_stop being overwritten")
        
        
        if interpolation == 'linear':
            time, momentum = self._linear_interpolation(mass, 
                                                        circumference, time, 
                                                        momentum, time_func)
            
        else:
            raise RuntimeError("Buggered")
        
        return time, momentum
            
                

    def _linear_interpolation(self, mass, circumference, time, momentum, 
                              time_func):
        
        pInit = np.interp(self.t_start, time, momentum)
        beta_0 = np.sqrt(1/(1 + (mass/pInit)**2))
        T0 = self._calc_t_rev(circumference, beta_0)

        nTurns = 0
        time_interp = [self.t_start]
        momentum_interp = [pInit]
        self.use_turns = [0]

        time_start_interp, time_end_interp = self._ramp_start_stop(time, 
                                                                   momentum)

        if time_end_interp > self.t_end:
            time_end_interp = self.t_end

        next_time = time_interp[0] + T0
        
        next_store_time = time_func(time_interp[0])

        while next_time < time_end_interp:
            
            next_momentum = np.interp(next_time, time, momentum)
            next_beta = np.sqrt(1/(1 + (mass/next_momentum)**2))
            next_time = next_time + self._calc_t_rev(circumference, next_beta)
            nTurns += 1

            if next_time >= next_store_time:
                time_interp.append(next_time)
                momentum_interp.append(next_momentum)
                self.use_turns.append(nTurns)
                next_store_time = time_func(time_interp[-1])
            
            if nTurns > self.n_turns:
                break
        else:
            if self.n_turns != np.inf:
                warnings.warn("Maximum time reached before number of turns")
            self.n_turns = nTurns
            
        return time_interp, momentum_interp



    def _calc_t_rev(self, circumference, beta):
        return circumference/(beta*cont.c)
    
    
    def _ramp_start_stop(self, time, momentum):
        
        time_start_ramp = np.max(time[momentum == momentum[0]])
        time_end_ramp = np.min(time[momentum == momentum[-1]])

        if time_start_ramp > self.t_start:
            time_start_ramp = self.t_start
        
        if time_end_ramp < self.t_end:
            time_end_ramp = self.t_end

        return time_start_ramp, time_end_ramp
    
    
def convert_data(synchronous_data, mass, charge,
                 synchronous_data_type='momentum', bending_radius=None):
        """ Function to convert synchronous data (i.e. energy program of the
        synchrotron) into momentum.

        Parameters
        ----------
        synchronous_data : float array
            The synchronous data to be converted to momentum
        mass : float or Particle.mass
            The mass of the particles in [eV/c**2]
        charge : int or Particle.charge
            The charge of the particles in units of [e]
        synchronous_data_type : str
            Type of input for the synchronous data ; can be 'momentum',
            'total energy', 'kinetic energy' or 'bending field' (last case
            requires bending_radius to be defined)
        bending_radius : float
            Bending radius in [m] in case synchronous_data_type is
            'bending field'

        Returns
        -------
        momentum : float array
            The input synchronous_data converted into momentum [eV/c]

        """

        if synchronous_data_type == 'momentum':
            momentum = synchronous_data
        elif synchronous_data_type == 'total energy':
            momentum = np.sqrt(synchronous_data**2 - mass**2)
        elif synchronous_data_type == 'kinetic energy':
            momentum = np.sqrt((synchronous_data+mass)**2 - mass**2)
        elif synchronous_data_type == 'bending field':
            if bending_radius is None:
                #InputDataError
                raise RuntimeError("ERROR in Ring: bending_radius is not " +
                                   "defined and is required to compute " +
                                   "momentum")
            momentum = synchronous_data*bending_radius*charge*c
        else:
            #InputDataError
            raise RuntimeError("ERROR in Ring: Synchronous data" +
                               " type not recognized!")

        return momentum


def load_data(filename, ignore=0, delimiter=None):
    r"""Helper function to load column-by-column data from a txt file to numpy
    arrays.

    Parameters
    ----------
    filename : str
        Name of the file containing the data.
    ignore : int
        Number of lines to ignore from the head of the file.
    delimiter : str
        Delimiting character between columns.

    Returns
    -------
    list of arrays
        Input data, column by column.

    """

    data = np.loadtxt(str(filename), skiprows=int(ignore),
                      delimiter=str(delimiter))

    return [np.ascontiguousarray(data[:, i]) for i in range(len(data[0]))]
