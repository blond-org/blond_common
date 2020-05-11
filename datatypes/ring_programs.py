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
    Base class for functions that are used to define ring parameters 
    (e.g. momentum, alpha)

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
    data_type : dict
        Dictionary containing relevant information to define the datatype
    timebase : str
        Either 'single', 'by_turn', or 'by_time' depending on the definition
        of the datatype.  As a string it is used as a key for the data_type 
        dict
    interpolation : str
        Identifier of the type of interpolation to be used when reshaping
        the array, currently only 'linear' has been implemented
    _sectioning : str
        identification of if the data defines a single machine section
        or multiple machine sections

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
        Prepare to combine multiple single sections datatype functions into a 
        single array.

        Parameters
        ----------
        cls : class
            The class of the final datatype.
        *args : datatypes
            The datatype arrays to be combined.
        interpolation : str, optional
            The interpolation method to be used for time dependent data, only
            used if not already defined in the data.  The default is None.
        **kwargs : keyword arguments
            None.

        Raises
        ------
        excpt.InputError
            If the passed data are not _ring_function type an InputError is
            raised.
            If the arrays are not defining a single section an InputError is
            raised.
        excpt.DataDefinitionError
            If the times of the input data are different and no interpolation
            type is available and DataDefinitionError is raised.
            If the function timebase attributes are not recognised a 
            DataDefinitionError is raised.

        Returns
        -------
        newArray : datatype
            An empty array to be populated with the combined data.
        timeBases : list of str
            The timebase identifiers of the data to be combined.
        use_times : list of int
            The times to be used for the combined data.
        use_turns : list of float
            The turn numbers to be used for the combined data.
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
                if a.timebase != 'single' and a.shape[1] < shortest:
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
        The data defining the programs
    time : iterable of floats
        The time array to be used for the data, to be used if the time 
        dependent data is not given as 2D array
    n_turns : int
        The number of turns to be used, if a single value is given it will be
        extended to n_turns length
    interpolation : str
        The type of interpolation to be used
    
    Attributes
    ----------
    data_type : dict
        Dictionary containing relevant information to define the datatype
    timebase : str
        Either 'single', 'by_turn', or 'by_time' depending on the definition
        of the datatype.  As a string it is used as a key for the data_type 
        dict
    interpolation : str
        Identifier of the type of interpolation to be used when reshaping
        the array, currently only 'linear' has been implemented
    _sectioning : str
        identification of if the data defines a single machine section
        or multiple machine sections
    """

    _conversions = {}
    """
    A dictionary storing the available classes for converting to.  Each
    key corresponds to the 'source' class attribute of the different 
    synchronous programs, and the value is the corresponding class.
    """

    def __new__(cls, *args, time = None, n_turns = None, interpolation = None):
        return super().__new__(cls, *args, time = time, n_turns = n_turns,
                               interpolation = interpolation)

    def to_momentum(self, inPlace = True, **kwargs):
        """
        Convert from any synchronous data to momentum either in place or as
        a new array.

        Parameters
        ----------
        inPlace : bool, optional
            If True the array will be changed in place, if False a new array
            is created and returned. The default is True.
        **kwargs : keyword arguments
            The keyword argumens requred for the conversion.  Requirements 
            depend on the data to be converted, but will include some of: 
                (bending_radius, rest_mass, charge).

        Returns
        -------
        datatype
            If inPlace is False a new datatype array is returned.
        """
        if self.source == 'momentum':
            return self._no_convert(inPlace)
        else:
            return self._convert('momentum', inPlace, **kwargs)

    def to_total_energy(self, inPlace = True, **kwargs):
        """
        Convert from any synchronous data to total energy either in place or
        as a new array.

        Parameters
        ----------
        inPlace : bool, optional
            If True the array will be changed in place, if False a new array
            is created and returned. The default is True.
        **kwargs : keyword arguments
            The keyword argumens requred for the conversion.  Requirements 
            depend on the data to be converted, but will include some of: 
                (bending_radius, rest_mass, charge).

        Returns
        -------
        datatype
            If inPlace is False a new datatype array is returned.
        """
        if self.source == 'energy':
            return self._no_convert(inPlace)
        else:
            return self._convert('energy', inPlace, **kwargs)

    def to_B_field(self, inPlace = True, **kwargs):
        """
        Convert from any synchronous data to magnetic field either in place or
        as a new array.

        Parameters
        ----------
        inPlace : bool, optional
            If True the array will be changed in place, if False a new array
            is created and returned. The default is True.
        **kwargs : keyword arguments
            The keyword argumens requred for the conversion.  Requirements 
            depend on the data to be converted, but will include some of: 
                (bending_radius, rest_mass, charge).

        Returns
        -------
        datatype
            If inPlace is False a new datatype array is returned.
        """

    def to_kin_energy(self, inPlace = True, **kwargs):
        """
        Convert from any synchronous data to kinetic energy either in place or
        as a new array.

        Parameters
        ----------
        inPlace : bool, optional
            If True the array will be changed in place, if False a new array
            is created and returned. The default is True.
        **kwargs : keyword arguments
            The keyword argumens requred for the conversion.  Requirements 
            depend on the data to be converted, but will include some of: 
                (bending_radius, rest_mass, charge).

        Returns
        -------
        datatype
            If inPlace is False a new datatype array is returned.
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
        Combine multiple single sections datatype functions into a 
        single array.

        Parameters
        ----------
        cls : class
            The class of the final datatype.
        *args : datatypes
            The datatype arrays to be combined.
        interpolation : str, optional
            The interpolation method to be used for time dependent data, only
            used if not already defined in the data.  The default is None.
        **kwargs : keyword arguments
            None.

        Raises
        ------
        excpt.InputError
            If the passed data are not _ring_function type an InputError is
            raised.

        Returns
        -------
        newArray : datatype
            The array containing the combined data.
        """
        if not all(isinstance(a, _synchronous_data_program) for a in args):
            raise excpt.InputError("Only _ring_function objects can be "
                                   + "combined")

        newArray, timeBases, use_times, use_turns \
            = cls._combine_single_sections(*args, 
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
        Preprocess the synchronous data to a full program for simulations or 
        calculations

        Parameters
        ----------
        mass : float
            Particle mass.
        circumference : float
            Ring circumference.
        interp_time : float or function, optional
            Defines the separation in time between points saved by the 
            interpolation. The default is None.
        interpolation : str, optional
            The type of interpolation to be used. The default is 'linear'.
        t_start : float, optional
            The earliest desired start time of the interpolation. The default 
            is 0.
        t_end : float, optional
            The latest desired end time of the interpolation. The default is 
            np.inf.
        flat_bottom : int, optional
            Number of turns of flat bottom.  Not yet implemented. The default 
            is 0.
        flat_top : int, optional
            Number of turns of flat top.  Not yet implemented. The default is 
            0.
        targetNTurns : int, optional
            The maximum desired number of turns. The default is np.inf.
        store_turns : bool, optional
            Flag to determine if the turn numbers will be included.  If True
            the program is interpolated turn by turn and the turn numbers are
            stored, if not no turn information is available.  Interpolation 
            without turn numbers is faster. The default is True.

        Raises
        ------
        excpt.DataDefinitionError
            If the data is not a momentum_program a DataDefinitionError is 
            raised.
        excpt.InputError
            If the requested type of interpolation is not available an 
            InputError is raised.

        Returns
        -------
        newArray : datatype
            The newly preprocessed array.  The data is of the form
            [turn numbers, time, synchronous data].  If multiple sections are
            defined additional synchronous data members are added for each 
            section.
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
        Convert from any synchronous data to momentum.

        Parameters
        ----------
        mass : float
            Particle mass in eV/c^2.
        charge : int, optional
            Charge in multiples of electron charge. The default is None.
        bending_radius : float, optional
            Bending radius of the machine. The default is None.
        inPlace : bool, optional
            Indicate of the array should be updated or a new array should be 
            returned. The default is True.

        Returns
        -------
        None or momentum_program
            If inPlace is True nothing is returned, if it is false a new
            momentum_program array is returned.

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
            return None
        
        else:
            return super().__new__(momentum_program, *newArray)


    def _convert_section(self, section, mass, charge = None, 
                         bending_radius = None):
        """
        Convert a single machine section to momentum.

        Parameters
        ----------
        section : int
            The section number to be converted.
        mass : float
            Particle mass in eV/c^2.
        charge : int, optional
            Charge in multiples of electron charge. The default is None.
        bending_radius : float, optional
            Bending radius of the machine. The default is None.

        Raises
        ------
        excpt.InputError
            If converting from magnetic field and bending_radius is not 
            supplied an InputError is raised.
        RuntimeError
            If the program is not convertible a RuntimeError is raised.

        Returns
        -------
        sectionFunction : array
            The converted data.
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
        Convert between different types of synchronous data, called by the
        to_momentum and equivalent functions if a conversion is necessary.

        Parameters
        ----------
        destination : str
            Identifies what the data will be converted to.
        inPlace : bool
            If True the data will be modified directly, if False a new array
            is created and returned.
        **kwargs : keyword arguments
            Keyword arguments needed for the conversion.  Requirements 
            depend on the data to be converted, but will include some of: 
                (bending_radius, rest_mass, charge).

        Returns
        -------
        None or _synchronous_data_program
            If inPlace is True nothing is returned.  If inPlace is false a new
            _synchronous_data_program is created and returned.
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
            
            self.__class__ = self._conversions[destination]
            return None

        else:
            return super().__new__(self._conversions[destination], *newArray)


    def _no_convert(self, inPlace):
        """
        Used by to_momentum and equivalents if the destination is the same as
        the source.

        Parameters
        ----------
        inPlace : bool
            If True nothing is done, if False a new duplicate array is created.

        Returns
        -------
        None or _sycnchronous_data_program
            If inPlace is True nothing is returned, if False a new duplicate
            array is created and returned.
        """
        if inPlace:
            return None
        else:
            return super().__new__(self.__class__, *self)


    def _time_from_turn(self, mass, circumference):
        """
        Used to acquire cumulative cycle time for data that has been defined
        by turn.

        Parameters
        ----------
        mass : float
            Particle mass in ev/c^2.
        circumference : float
            Machine circumference.

        Returns
        -------
        array of floats
            Cycle times of each turn.

        """
        trev = rt.mom_to_trev(self[0], mass, circ=circumference)
        return np.cumsum(trev)


    def _linear_interpolation_no_turns(self, mass, circumference, time, 
                                       section):
        """
        Linearly interpolate the cycle without considering turn numbers.

        Parameters
        ----------
        mass : float
            Particle mass in ev/c^2.
        circumference : float
            Machine circumference.
        time : tuple of (function, float, float)
            (interp_time function, start time, stop time) used for the 
            interpolation.
        section : int
            Section number.

        Raises
        ------
        RuntimeError
            If interp_time function returns 0 an infinite loop would be 
            entered and a RuntimeError is raised.

        Returns
        -------
        np.NaN
            In place of the total number of turns, np.NaN is returned.
        array of np.NaN
            In place of the turn numbers an array of np.NaN is returned.
        array of floats
            The times of the interpolated points.
        array of floats
            The momentum at the interpolated points.
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
        Linearly interpolate the synchronous data including the turn numbers.

        Parameters
        ----------
        mass : float
            Particle mass.
        circumference : float
            Ring circumference.
        time : tuple of (function, float, float)
            (interp_time function, start time, stop time) used for the 
            interpolation.
        targetNTurns : int, optional
            The maximum desired number of turns. The default is np.inf.
        section : int
            Section number.

        Returns
        -------
        int
            The total number of turns.
        array of int
            The turn numbers.
        array of floats
            The times of the interpolated points.
        array of floats
            The momentum at the interpolated points.
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
        Interpolate the synchronous data maintaining a linearly varying first
        derivative.

        Parameters
        ----------
        mass : float
            Particle mass.
        circumference : float
            Ring circumference.
        time : tuple of (function, float, float)
            (interp_time function, start time, stop time) used for the 
            interpolation.
        targetNTurns : int, optional
            The maximum desired number of turns. The default is np.inf.
        section : int
            Section number.

        Returns
        -------
        int
            The total number of turns.
        array of int
            The turn numbers.
        array of floats
            The times of the interpolated points.
        array of floats
            The momentum at the interpolated points.
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

        #TODO: adjust respecting the correct boundaries
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
        Get the start and end times of the ramp.

        Raises
        ------
        RuntimeError
            If the synchronous data is not defined by time a RuntimeError is
            raised.

        Returns
        -------
        time_start_ramp : float
            The time that the ramp starts.
        time_end_ramp : float
            The time that the ramp stops.
        """
        if self.timebase != 'by_time':
            raise RuntimeError("Only implemented for by_time functions")
        
        time_start_ramp = np.max(self[0, 0][self[0, 1] == self[0, 1, 0]])
        time_end_ramp = np.max(self[0, 0][self[0, 1] == self[0, 1, -1]])

        return time_start_ramp, time_end_ramp

    @classmethod
    def _add_to_conversions(cls):
        """
        Add the class to the _conversions class attribute dict.

        Parameters
        ----------
        cls : class
            The available class.
        """
        cls._conversions[cls.source] = cls


class momentum_program(_synchronous_data_program):
    """
    Class for  defining synchronous data as momentum.

    Parameters
    ----------
    *args : float, 1D iterable of floats, 2D iterable of floats
        The data defining the programs
    time : iterable of floats
        The time array to be used for the data, to be used if the time 
        dependent data is not given as 2D array
    n_turns : int
        The number of turns to be used, if a single value is given it will be
        extended to n_turns length
    interpolation : str
        The type of interpolation to be used
    
    Attributes
    ----------
    data_type : dict
        Dictionary containing relevant information to define the datatype
    timebase : str
        Either 'single', 'by_turn', or 'by_time' depending on the definition
        of the datatype.  As a string it is used as a key for the data_type 
        dict
    interpolation : str
        Identifier of the type of interpolation to be used when reshaping
        the array, currently only 'linear' has been implemented
    _sectioning : str
        identification of if the data defines a single machine section
        or multiple machine sections
    """
    source = 'momentum'
    """A str identifying the data"""

class total_energy_program(_synchronous_data_program):
    """
    Class for  defining synchronous data as total energy.

    Parameters
    ----------
    *args : float, 1D iterable of floats, 2D iterable of floats
        The data defining the programs
    time : iterable of floats
        The time array to be used for the data, to be used if the time 
        dependent data is not given as 2D array
    n_turns : int
        The number of turns to be used, if a single value is given it will be
        extended to n_turns length
    interpolation : str
        The type of interpolation to be used
    
    Attributes
    ----------
    data_type : dict
        Dictionary containing relevant information to define the datatype
    timebase : str
        Either 'single', 'by_turn', or 'by_time' depending on the definition
        of the datatype.  As a string it is used as a key for the data_type 
        dict
    interpolation : str
        Identifier of the type of interpolation to be used when reshaping
        the array, currently only 'linear' has been implemented
    _sectioning : str
        identification of if the data defines a single machine section
        or multiple machine sections
    """
    source = 'energy'
    """A str identifying the data"""

class kinetic_energy_program(_synchronous_data_program):
    """
    Class for  defining synchronous data as kinetic energy.

    Parameters
    ----------
    *args : float, 1D iterable of floats, 2D iterable of floats
        The data defining the programs
    time : iterable of floats
        The time array to be used for the data, to be used if the time 
        dependent data is not given as 2D array
    n_turns : int
        The number of turns to be used, if a single value is given it will be
        extended to n_turns length
    interpolation : str
        The type of interpolation to be used
    
    Attributes
    ----------
    data_type : dict
        Dictionary containing relevant information to define the datatype
    timebase : str
        Either 'single', 'by_turn', or 'by_time' depending on the definition
        of the datatype.  As a string it is used as a key for the data_type 
        dict
    interpolation : str
        Identifier of the type of interpolation to be used when reshaping
        the array, currently only 'linear' has been implemented
    _sectioning : str
        identification of if the data defines a single machine section
        or multiple machine sections
    """
    source = 'kin_energy'
    """A str identifying the data"""

class bending_field_program(_synchronous_data_program):
    """
    Class for  defining synchronous data as magnetic field.

    Parameters
    ----------
    *args : float, 1D iterable of floats, 2D iterable of floats
        The data defining the programs
    time : iterable of floats
        The time array to be used for the data, to be used if the time 
        dependent data is not given as 2D array
    n_turns : int
        The number of turns to be used, if a single value is given it will be
        extended to n_turns length
    interpolation : str
        The type of interpolation to be used
    
    Attributes
    ----------
    data_type : dict
        Dictionary containing relevant information to define the datatype
    timebase : str
        Either 'single', 'by_turn', or 'by_time' depending on the definition
        of the datatype.  As a string it is used as a key for the data_type 
        dict
    interpolation : str
        Identifier of the type of interpolation to be used when reshaping
        the array, currently only 'linear' has been implemented
    _sectioning : str
        identification of if the data defines a single machine section
        or multiple machine sections
    """
    source = 'B_field'
    """A str identifying the data"""

for data in [momentum_program, total_energy_program, kinetic_energy_program,
             bending_field_program]:
    data._add_to_conversions()


class momentum_compaction(_ring_function):
    """
    Class defining momentum_compaction factors
    
    Parameters
    ----------
    *args : float, 1D iterable of floats, 2D iterable of floats
        The data defining the programs
    order : int
        The order of the momentum compaction factor program
    time : iterable of floats
        The time array to be used for the data, to be used if the time 
        dependent data is not given as 2D array
    n_turns : int
        The number of turns to be used, if a single value is given it will be
        extended to n_turns length
    interpolation : str
        The type of interpolation to be used
    
    Attributes
    ----------
    data_type : dict
        Dictionary containing relevant information to define the datatype
    timebase : str
        Either 'single', 'by_turn', or 'by_time' depending on the definition
        of the datatype.  As a string it is used as a key for the data_type 
        dict
    interpolation : str
        Identifier of the type of interpolation to be used when reshaping
        the array, currently only 'linear' has been implemented
    _sectioning : str
        identification of if the data defines a single machine section
        or multiple machine sections
    order : int
        The order of the momentum compaction factor program
    """
    def __new__(cls, *args, order = 0, time = None, n_turns = None, 
                interpolation = 'linear'):

        return super().__new__(cls, *args, time = time, 
                                 n_turns = n_turns, allow_single = True,
                                 interpolation = 'linear', order=order)

    @classmethod
    def combine_single_sections(cls, *args, interpolation = None):
        """
        Combine multiple single sections into a single array.

        Parameters
        ----------
        cls : class
            The class of the final datatype.
        *args : datatypes
            The datatype arrays to be combined.
        interpolation : str, optional
            The interpolation method to be used for time dependent data, only
            used if not already defined in the data.  The default is None.

        Raises
        ------
        excpt.InputError
            If the passed data are not momentum_compaction type an InputError 
            is raised.
            If the passed data do not all have the same order an InputError is
            raised.

        Returns
        -------
        newArray : datatype
            The new array combining all of the passed sections.
        """
        if not all(isinstance(a, momentum_compaction) for a in args):
            raise excpt.InputError("Only momentum_compaction objects can be "
                                   + "combined")

        if not all(a.order == args[0].order for a in args):
            raise excpt.InputError("Only programs with equal order can be "
                                   + "combined")

        newArray, timeBases, use_times, use_turns \
            = cls._combine_single_sections(*args, 
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
