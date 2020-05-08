#General imports
import numpy as np
import sys
import os
import warnings
import scipy.constants as cont
import matplotlib.pyplot as plt

#Common imports
from ..devtools import exceptions as excpt
from ..devtools import assertions as assrt
from ..utilities import rel_transforms as rt
from . import blond_function as bf
from ._core import _function, _expand_function, _check_time_turns,\
                   _get_dats_types, _check_data_types, _expand_singletons,\
                   _check_turn_numbers


class _ring_function(_function):
    """
    
    Parameters
    ----------
    *args : float, 1D iterable of floats, 2D iterable of floats
    time : iterable of floats
    n_turns : int
    allow_single : bool
    interpolation : str
    **kwargs : keyword arguments
    
    Attributes
    ----------
    As _function class plus:
        _sectioning : str
        
    """
    def __new__(cls, *args, time = None, n_turns = None, 
                allow_single = False, interpolation = None, **kwargs):
        args = _expand_function(*args)
        _check_time_turns(time, n_turns)
            
        data_points, data_types = _get_dats_types(*args, time = time, \
                                                  n_turns = n_turns)

        _check_data_types(data_types, allow_single)
        if allow_single:
            data_types, data_points = _expand_singletons(data_types, 
                                                         data_points)

        if 'by_turn' in data_types:
            _check_turn_numbers(data_points, data_types)
        
        
        if len(data_types) == 1:
            data_type = {'timebase': data_types[0], 
                         'sectioning': 'single_section'}
        else:
            data_type = {'timebase': data_types[0], 
                         'sectioning': 'multi_section'}
        
        data_type = {**data_type, **kwargs}
        
        return super().__new__(cls, data_points, data_type, interpolation)
    
    
    @classmethod
    def _combine_single_sections(cls, *args,
                                interpolation = None, **kwargs):
        """
        

        Parameters
        ----------
        cls : TYPE
            DESCRIPTION.
        *args : TYPE
            DESCRIPTION.
        interpolation : TYPE, optional
            DESCRIPTION. The default is None.
        **kwargs : TYPE
            DESCRIPTION.

        Raises
        ------
        excpt
            DESCRIPTION.

        Returns
        -------
        newArray : TYPE
            DESCRIPTION.
        timeBases : TYPE
            DESCRIPTION.
        use_times : TYPE
            DESCRIPTION.
        use_turns : TYPE
            DESCRIPTION.

        """
        if not all(isinstance(a, _ring_function) for a in args):
            raise excpt.InputError("Only _ring_function objects can be "
                                   + "combined")

        timeBases = [a.timebase for a in args]
        try:
            assrt.equal_arrays(*timeBases, msg = 'Attempting to combine '
                                                 + 'sections with different '
                                                 + 'timebases',
                                                 exception = excpt.InputError)
        except:
            turns = len([t for t in timeBases if t == 'by_turn']) == 0
            times = len([t for t in timeBases if t == 'by_time']) == 0
            if turns == times:
                raise

        if any(a.sectioning != 'single_section' for a in args):
            raise excpt.InputError("Only single section functions can be "
                                   + "combined")

        nFuncs = len(args)
        if 'by_time' in timeBases[0]:
            input_times = []
            for a in args:
                try:
                    input_times += a[0,0].tolist()
                except IndexError:
                    pass
            use_times = sorted(set(input_times))

            try:
                assrt.equal_arrays(*(a[0, 0] for a in args 
                                                 if a.timebase != 'single'),
                                   msg=None,
                                   exception = excpt.InputError)
            except excpt.InputError:
                if interpolation is not None:
                    for a in args:
                        a.interpolation = interpolation
                if any(a.interpolation is None for a in args 
                                                   if a.timebase != 'single'):
                    raise excpt.DataDefinitionError("Combining functions with "
                                                    + "different time axes "
                                                    + "requires interpolation "
                                                    + "method to be defined.")
            use_turns = None
            newArray = cls.zeros([nFuncs, 2, len(use_times)])
        elif 'by_turn' in timeBases:
            
            try:
                assrt.equal_array_lengths(*args, msg = None,
                                          exception = excpt.InputError)
            except excpt.InputError:
                warnings.warn("Arrays cover different numbers of turns, "
                              + "shortest array length will be used")
            
            shortest = np.inf
            for a in args:
                if a.timebase is not 'single' and a.shape[1] < shortest:
                    shortest = a.shape[1]
            use_turns = np.arange(shortest).astype(int)
            use_times = None
            newArray = cls.zeros([nFuncs, len(use_turns)])
        elif timeBases[0] == 'single':
            use_turns = None
            use_times = None
            newArray = cls.zeros(nFuncs)
        else:
            raise excpt.DataDefinitionError("function timebases not "
                                            + "recognised")
        
        return newArray, timeBases, use_times, use_turns
    
    
    
    @property
    def sectioning(self):
        """
        Get or set the sectioning.  Setting the sectioning will also update
        the data_type dict.
        """
        try:
            return self._sectioning
        except AttributeError:
            return None
    
    @sectioning.setter
    def sectioning(self, value):
        self._check_data_type('sectioning', value)
        self._sectioning = value


class _synchronous_data_program(_ring_function):
    """
    Base class for momentum-like ring programs (momentum, B-field, etc).

    Parameters
    ----------
    *args : float, 1D iterable of floats, 2D iterable of floats
    time : iterable of floats
    n_turns : int
    interpolation : str
    
    Attributes
    ----------
    As _ring_function class plus
    _sectioning : str
    """

    conversions = {}

    def __new__(cls, *args, time = None, n_turns = None, interpolation = None):
        return super().__new__(cls, *args, time = time, n_turns = n_turns,
                               interpolation = interpolation)

    def to_momentum(self, inPlace = True, **kwargs):
        """
        

        Parameters
        ----------
        inPlace : TYPE, optional
            DESCRIPTION. The default is True.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if self.source == 'momentum':
            return self._no_convert(inPlace)
        else:
            return self._convert('momentum', inPlace, **kwargs)

    def to_total_energy(self, inPlace = True, **kwargs):
        """
        

        Parameters
        ----------
        inPlace : TYPE, optional
            DESCRIPTION. The default is True.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if self.source == 'energy':
            return self._no_convert(inPlace)
        else:
            return self._convert('energy', inPlace, **kwargs)

    def to_B_field(self, inPlace = True, **kwargs):
        """
        

        Parameters
        ----------
        inPlace : TYPE, optional
            DESCRIPTION. The default is True.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if self.source == 'B_field':
            return self._no_convert(inPlace)
        else:
            return self._convert('B_field', inPlace, **kwargs)

    def to_kin_energy(self, inPlace = True, **kwargs):
        """
        

        Parameters
        ----------
        inPlace : TYPE, optional
            DESCRIPTION. The default is True.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if self.source == 'kin_energy':
            return self._no_convert(inPlace)
        else:
            return self._convert('kin_energy', inPlace, **kwargs)   


    #TODO: Consider multi-section version
    @classmethod
    def combine_single_sections(cls, *args,
                                interpolation = None, **kwargs):
        """
        

        Parameters
        ----------
        cls : TYPE
            DESCRIPTION.
        *args : TYPE
            DESCRIPTION.
        interpolation : TYPE, optional
            DESCRIPTION. The default is None.
        **kwargs : TYPE
            DESCRIPTION.

        Raises
        ------
        excpt
            DESCRIPTION.

        Returns
        -------
        newArray : TYPE
            DESCRIPTION.

        """
        if not all(isinstance(a, _synchronous_data_program) for a in args):
            raise excpt.InputError("Only _ring_function objects can be "
                                   + "combined")

        newArray, timeBases, use_times, use_turns \
            = super()._combine_single_sections(*args, 
                                               interpolation = interpolation, 
                                               **kwargs)
            
        bending_radius = kwargs.pop('bending_radius', None)
        if not hasattr(bending_radius, '__iter__'):
            bending_radius = [bending_radius]*newArray.shape[0]
        for i, (a, b) in enumerate(zip(args, bending_radius)):
            if timeBases[0] != 'single':
                section = a.reshape(use_time = use_times, 
                                    use_turns = use_turns)
            else:
                section = a.copy()
            if cls.source != a.source:
                section._convert(cls.source, inPlace = True, 
                                 bending_radius = b, **kwargs)
            newArray[i] = section
        
        newArray.timebase = timeBases[0]
        if newArray.shape[0] > 1:
            newArray.sectioning = 'multi_section'
        else:
            newArray.sectioning = 'single_section'
        
        newArray.interpolation = interpolation
        
        return newArray


    #TODO: multi-section
    #TODO: handle flat_bottom and flat_top
    def preprocess(self, mass, circumference, interp_time = None, 
                   interpolation = 'linear', t_start = 0, t_end = np.inf,
                   flat_bottom = 0, flat_top = 0, targetNTurns = np.inf,
                   store_turns = True):
        """
        

        Parameters
        ----------
        mass : TYPE
            DESCRIPTION.
        circumference : TYPE
            DESCRIPTION.
        interp_time : TYPE, optional
            DESCRIPTION. The default is None.
        interpolation : TYPE, optional
            DESCRIPTION. The default is 'linear'.
        t_start : TYPE, optional
            DESCRIPTION. The default is 0.
        t_end : TYPE, optional
            DESCRIPTION. The default is np.inf.
        flat_bottom : TYPE, optional
            DESCRIPTION. The default is 0.
        flat_top : TYPE, optional
            DESCRIPTION. The default is 0.
        targetNTurns : TYPE, optional
            DESCRIPTION. The default is np.inf.
        store_turns : TYPE, optional
            DESCRIPTION. The default is True.

        Raises
        ------
        excpt
            DESCRIPTION.

        Returns
        -------
        newArray : TYPE
            DESCRIPTION.

        """
        if not isinstance(self, momentum_program):
            raise excpt.DataDefinitionError("Only momentum functions "
                                                 + "can be preprocessed, not "
                                                 + self.__class__.__name__ 
                                                 + ", first run " 
                                                 + self.__class__.__name__ 
                                                 + ".convert")

        interp_funcs = {'linear': self._linear_interpolation,
                        'derivative': self._derivative_interpolation}
        
        if interpolation not in interp_funcs:
            raise excpt.InputError(f"Available interpolation options are:\
                                        {tuple(interp_funcs.keys())}")

        if not hasattr(interp_time, '__call__'):
            if interp_time is None:
                _interp_time = 0
            else:
                _interp_time = interp_time
            interp_time = lambda x: x + _interp_time
        
        if self.timebase == 'by_time':
            if t_start < self[0, 0, 0]:
                warnings.warn("t_start too early, starting from " 
                              + str(self[0, 0, 0]))
                t_start = self[0, 0, 0]
    
            if t_end > self[0, 0, -1]:
                warnings.warn("t_stop too late, ending at " 
                              + str(self[0, 0, -1]))
                t_end = self[0, 0, -1]
        #TODO: Treat derivative interpolation without storing turns
            for s in range(self.shape[0]):
                if store_turns:
                    nTurns, useTurns, time, momentum \
                                = interp_funcs[interpolation](mass,
                                                              circumference, 
                                                              (interp_time, 
                                                               t_start, t_end), 
                                                              targetNTurns, s)
                else:
                    nTurns, useTurns, time, momentum \
                        = self._linear_interpolation_no_turns(mass, 
                                                              circumference, 
                                                              (interp_time,
                                                               t_start, t_end),
                                                              s)
                    
        #TODO: Sampling with turn by turn data
        #TODO: nTurns != self.shape[1]
        elif self.timebase == 'by_turn':
            if targetNTurns < np.inf:
                nTurns = targetNTurns
            else:
                nTurns = self.shape[1]
            useTurns = np.arange(nTurns)
            time = self._time_from_turn(mass, circumference)
            momentum = self[0]
        
        #TODO: Handle passed number of turns
        elif self.timebase == 'single':
            time = [0]
            nTurns = 1
            useTurns = [0]
            momentum = self.copy()

        newArray = np.zeros([2+self.shape[0], len(useTurns)])
        newArray[0, :] = useTurns
        newArray[1, :] = time
        
        for s in range(self.shape[0]):
            newArray[s+2] = momentum
            
        newArray = newArray.view(momentum_program)
        
        newArray.n_turns = nTurns
        
        return newArray


    def convert(self, mass, charge = None, bending_radius = None, 
                inPlace = True):
        """
        

        Parameters
        ----------
        mass : TYPE
            DESCRIPTION.
        charge : TYPE, optional
            DESCRIPTION. The default is None.
        bending_radius : TYPE, optional
            DESCRIPTION. The default is None.
        inPlace : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        newArray = np.zeros(self.shape)

        for s in range(self.shape[0]):
            if self.timebase == 'by_time':
                newArray[s, 1] = self._convert_section(s, mass, charge,
                                                        bending_radius)
                newArray[s, 0] = self[s, 0]
            else:
                newArray[s] = self._convert_section(s, mass, charge,
                                                     bending_radius)

        if inPlace:
            for s in range(self.shape[0]):
                if self.timebase == 'by_time':
                    self[s, 1] = newArray[s, 1]
                else:
                    self[s] = newArray[s]
            
            self.__class__ = momentum_program
        
        else:
            return super().__new__(momentum_program, *newArray)


    def _convert_section(self, section, mass, charge = None, 
                         bending_radius = None):
        """
        

        Parameters
        ----------
        section : TYPE
            DESCRIPTION.
        mass : TYPE
            DESCRIPTION.
        charge : TYPE, optional
            DESCRIPTION. The default is None.
        bending_radius : TYPE, optional
            DESCRIPTION. The default is None.

        Raises
        ------
        excpt
            DESCRIPTION.
        RuntimeError
            DESCRIPTION.

        Returns
        -------
        sectionFunction : TYPE
            DESCRIPTION.

        """
        if self.timebase == 'by_time':
            sectionFunction = np.array(self[section, 1])
        else:
            sectionFunction = np.array(self[section])
        
        if isinstance(self, momentum_program):
            pass
        elif isinstance(self, total_energy_program):
            sectionFunction = rt.energy_to_momentum(sectionFunction, mass)
        elif isinstance(self, kinetic_energy_program):
            sectionFunction = rt.kin_energy_to_momentum(sectionFunction, mass)
        elif isinstance(self, bending_field_program):
            if None in (bending_radius, charge):
                raise excpt.InputError("Converting from bending field "
                                            + "requires both charge and "
                                            + "bending radius to be defined")
            sectionFunction = rt.B_field_to_momentum(sectionFunction, 
                                                     bending_radius, 
                                                     charge)
    
        else:
            raise RuntimeError("Function type invalid")

        return sectionFunction
    
    
    def _convert(self, destination, inPlace, **kwargs):
        """
        

        Parameters
        ----------
        destination : TYPE
            DESCRIPTION.
        inPlace : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        conversion_function = getattr(rt, self.source + '_to_' + destination)
        newArray = np.zeros(self.shape)
        
        arguments = conversion_function.__code__.co_varnames[1:-1]
        arguments = {arg: kwargs.pop(arg, None) for arg in arguments}        

        massTypes = ['rest_mass', 'n_nuc', 'atomic_mass']

        checkList = tuple(arguments[arg] for arg in arguments \
                         if arg not in massTypes)
        
        checkKwargs = tuple(arg for arg in arguments \
                        if arg not in massTypes)
        
        errorMsg = 'conversion from ' + self.source + ' to ' + destination \
                   + ' requires all of ' + str(checkKwargs) + ' to be defined'
        
        if all(m in arguments for m in massTypes):
            errorMsg += ' and one of (rest_mass, n_nuc, atomic_mass)'
            assrt.single_not_none(*(arguments[m] for m in massTypes), 
                                  msg = errorMsg, 
                                  exception = excpt.InputError)
        
        assrt.all_not_none(*checkList, msg = errorMsg, 
                           exception = excpt.InputError)
        
        for s in range(self.shape[0]):
            if self.timebase == 'by_time':
                newArray[s, 1] = conversion_function(self[s, 1], **arguments)
                newArray[s, 0] = self[s, 0]
            else:
                newArray[s] = conversion_function(self[s], **arguments)

        if inPlace:
            for s in range(self.shape[0]):
                if self.timebase == 'by_time':
                    self[s, 1] = newArray[s, 1]
                else:
                    self[s] = newArray[s]
            
            self.__class__ = self.conversions[destination]

        else:
            return super().__new__(self.conversions[destination], *newArray)


    def _no_convert(self, inPlace):
        """
        

        Parameters
        ----------
        inPlace : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if inPlace:
            return self
        else:
            return super().__new__(self.__class__, *self)


    def _time_from_turn(self, mass, circumference):
        """
        

        Parameters
        ----------
        mass : TYPE
            DESCRIPTION.
        circumference : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        trev = rt.mom_to_trev(self[0], mass, circ=circumference)
        return np.cumsum(trev)


    def _linear_interpolation_no_turns(self, mass, circumference, time, 
                                       section):
        """
        

        Parameters
        ----------
        mass : TYPE
            DESCRIPTION.
        circumference : TYPE
            DESCRIPTION.
        time : TYPE
            DESCRIPTION.
        section : TYPE
            DESCRIPTION.

        Raises
        ------
        RuntimeError
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.
        momentum_interp : TYPE
            DESCRIPTION.

        """
        time_func = time[0]
        start = time[1]
        stop = time[2]

        interp_time = [start]

        while interp_time[-1] < stop:
            next_time = time_func(interp_time[-1])
            if not next_time > interp_time[-1]:
                raise RuntimeError("Attempting to interpolate with 0 "
                                   + "sample spacing")
            else:
                interp_time.append(next_time)

        if interp_time[-1] > stop:
            interp_time = interp_time[:-1]
        
        input_time = self[section, 0]
        input_momentum = self[section, 1]
        
        momentum_interp = np.interp(interp_time, input_time, input_momentum)
        
        return (np.NaN, np.full(len(interp_time), np.NaN), 
                np.array(interp_time), momentum_interp)


    def _linear_interpolation(self, mass, circumference, time, targetNTurns,
                              section):
        """
        

        Parameters
        ----------
        mass : TYPE
            DESCRIPTION.
        circumference : TYPE
            DESCRIPTION.
        time : TYPE
            DESCRIPTION.
        targetNTurns : TYPE
            DESCRIPTION.
        section : TYPE
            DESCRIPTION.

        Returns
        -------
        nTurns : TYPE
            DESCRIPTION.
        use_turns : TYPE
            DESCRIPTION.
        time_interp : TYPE
            DESCRIPTION.
        momentum_interp : TYPE
            DESCRIPTION.

        """
        time_func = time[0]
        start = time[1]
        stop = time[2]

        pInit = np.interp(start, self[section, 0], self[section, 1])
        beta_0 = rt.mom_to_beta(pInit, mass)
        T0 = rt.beta_to_trev(beta_0, circumference)

        nTurns = 0
        time_interp = [start]
        momentum_interp = [pInit]
        use_turns = [0]

        next_time = time_interp[0] + T0
        
        next_store_time = time_func(time_interp[0])

        input_time = self[section, 0].tolist()
        input_momentum = self[section, 1].tolist()

        k = 0

        while next_time < stop:
            while next_time > input_time[k]:
                k += 1

            next_momentum = input_momentum[k-1] \
                            + (input_momentum[k] - input_momentum[k-1]) \
                             * (next_time - input_time[k-1]) \
                             / (input_time[k] - input_time[k-1])
                             
            next_beta = rt.mom_to_beta(next_momentum, mass)
            next_time = next_time + rt.beta_to_trev(next_beta, circumference)
            nTurns += 1

            if next_time >= next_store_time:
                time_interp.append(next_time)
                momentum_interp.append(next_momentum)
                use_turns.append(nTurns)
                next_store_time = time_func(time_interp[-1])

            if nTurns >= targetNTurns-1:
                break

        else:
            if targetNTurns != np.inf:
                warnings.warn("Maximum time reached before number of turns")

        return nTurns, use_turns, time_interp, momentum_interp

    def _derivative_interpolation(self, mass, circumference, time, 
                                  targetNTurns, section):
        """
        

        Parameters
        ----------
        mass : TYPE
            DESCRIPTION.
        circumference : TYPE
            DESCRIPTION.
        time : TYPE
            DESCRIPTION.
        targetNTurns : TYPE
            DESCRIPTION.
        section : TYPE
            DESCRIPTION.

        Returns
        -------
        nTurns : TYPE
            DESCRIPTION.
        use_turns : TYPE
            DESCRIPTION.
        time_interp : TYPE
            DESCRIPTION.
        momentum_interp : TYPE
            DESCRIPTION.

        """
        time_func = time[0]
        start = time[1]
        stop = time[2]
        
        #TODO: Is it acceptable to have linear interp here?
        pInit = np.interp(start, self[section, 0], self[section, 1])
        beta_0 = rt.mom_to_beta(pInit, mass)
        T0 = rt.beta_to_trev(beta_0, circumference)

        nTurns = 0
        time_interp = [start]
        momentum_interp = [pInit]
        use_turns = [0]

        next_time = time_interp[0] + T0
        
        next_store_time = time_func(time_interp[0])

        input_time = self[section, 0].tolist()
        input_momentum = self[section, 1].tolist()

        k = 0

        
        momentum_initial = momentum_interp[0]
        #TODO: Compare gradients with other methods of derivative calculation
        momentum_derivative = np.gradient(input_momentum)\
                                /np.gradient(input_time)

        momentum_derivative_interp = [momentum_derivative[0]]
        next_momentum = momentum_initial
        next_beta = rt.mom_to_beta(next_momentum, mass)
        while next_time < stop:
            while next_time > input_time[k]:
                k += 1

            derivative_point = momentum_derivative[k-1] \
                            + (momentum_derivative[k] \
                               - momentum_derivative[k-1]) \
                             * (next_time - input_time[k-1]) \
                             / (input_time[k] - input_time[k-1])
                             
            momentum_derivative_interp.append(derivative_point)
            future_time = next_time + rt.beta_to_trev(next_beta, circumference)
            next_momentum += (future_time - next_time) \
                * derivative_point

            next_beta = rt.mom_to_beta(next_momentum, mass)
            next_time = future_time
            nTurns += 1

            if next_time >= next_store_time:
                time_interp.append(next_time)
                momentum_interp.append(next_momentum)
                use_turns.append(nTurns)
                next_store_time = time_func(time_interp[-1])

            if nTurns >= targetNTurns-1:
                break

        else:
            if targetNTurns != np.inf:
                warnings.warn("Maximum time reached before number of turns")

        # Adjust result to get flat top energy correct as derivation and
        # integration leads to ~10^-8 error in flat top momentum
        # momentum_interp = np.asarray(momentum_interp)
        # momentum_interp -= momentum_interp[0]
        # momentum_interp /= momentum_interp[-1]
        # momentum_interp *= input_momentum[-1] - input_momentum[0]
        # momentum_interp += input_momentum[0]

        return nTurns, use_turns, time_interp, momentum_interp
        
        
    def _ramp_start_stop(self):
        """
        

        Raises
        ------
        RuntimeError
            DESCRIPTION.

        Returns
        -------
        time_start_ramp : TYPE
            DESCRIPTION.
        time_end_ramp : TYPE
            DESCRIPTION.

        """
        if self.timebase != 'by_time':
            raise RuntimeError("Only implemented for by_time functions")
        
        time_start_ramp = np.max(self[0, 0][self[0, 1] == self[0, 1, 0]])
        time_end_ramp = np.max(self[0, 0][self[0, 1] == self[0, 1, -1]])

        return time_start_ramp, time_end_ramp

    @classmethod
    def _add_to_conversions(cls):
        """
        

        Parameters
        ----------
        cls : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        cls.conversions[cls.source] = cls


class momentum_program(_synchronous_data_program):
    source = 'momentum'
    """A str identifying the data"""

class total_energy_program(_synchronous_data_program):
    source = 'energy'
    """A str identifying the data"""

class kinetic_energy_program(_synchronous_data_program):
    source = 'kin_energy'
    """A str identifying the data"""

class bending_field_program(_synchronous_data_program):
    source = 'B_field'
    """A str identifying the data"""

for data in [momentum_program, total_energy_program, kinetic_energy_program,
             bending_field_program]:
    data._add_to_conversions()


class momentum_compaction(_ring_function):
    """
    Class dedicated to momentum_compaction factors
    
    Parameters
    ----------
    *args : float, 1D iterable of floats, 2D iterable of floats
    order : int
    time : iterable of floats
    n_turns : int
    interpolation : str
    
    Attributes
    ----------
    As _ring_function plus
    order : int
    """
    def __new__(cls, *args, order = 0, time = None, n_turns = None, 
                interpolation = 'linear'):

        return super().__new__(cls, *args, time = time, 
                                 n_turns = n_turns, allow_single = True,
                                 interpolation = 'linear', order=order)

    @classmethod
    def combine_single_sections(cls, *args, interpolation = None):
        """
        

        Parameters
        ----------
        cls : TYPE
            DESCRIPTION.
        *args : TYPE
            DESCRIPTION.
        interpolation : TYPE, optional
            DESCRIPTION. The default is None.

        Raises
        ------
        excpt
            DESCRIPTION.

        Returns
        -------
        newArray : TYPE
            DESCRIPTION.

        """
        if not all(isinstance(a, momentum_compaction) for a in args):
            raise excpt.InputError("Only momentum_compaction objects can be "
                                   + "combined")

        if not all(a.order == args[0].order for a in args):
            raise excpt.InputError("Only programs with equal order can be "
                                   + "combined")

        newArray, timeBases, use_times, use_turns \
            = super()._combine_single_sections(*args, 
                                               interpolation = interpolation)
            
        for i, a in enumerate(args):
            if timeBases[0] != 'single':
                section = a.reshape(use_time = use_times, 
                                    use_turns = use_turns)
            else:
                section = a.copy()
            newArray[i] = section
        
        newArray.timebase = timeBases[0]
        if newArray.shape[0] > 1:
            newArray.sectioning = 'multi_section'
        else:
            newArray.sectioning = 'single_section'
        
        newArray.interpolation = interpolation
        
        return newArray


    @property
    def order(self):
        """
        Get or set the order.  Setting the order will also update
        the data_type dict.
        """
        try:
            return self._order
        except AttributeError:
            return None
    
    @order.setter
    def order(self, value):
        self._check_data_type('order', value)
        self._order = value
