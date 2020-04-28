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

#TODO: Overwrite some np funcs (e.g. __iadd__) where necessary
#TODO: In derived classes handle passing datatype as input
#TODO: Check for ragged array
class _function(np.ndarray):
    
    def __new__(cls, input_array, data_type=None, interpolation = None):
        
        if data_type is None:
            raise excpt.InputError("data_type must be specified")
        
        try:
            obj = np.asarray(input_array).view(cls)
        except ValueError:
            raise excpt.InputError("Function components could not be " \
                                        + "correctly coerced into ndarray, " \
                                        + "check input dimensionality")
        
        obj.data_type = data_type
        
        obj.interpolation = interpolation
        
        return obj
    
    def __array_finalize__(self, obj):

        if obj is None:
            return

        self.data_type = getattr(obj, 'data_type', None)


    @classmethod
    def zeros(cls, shape, data_type = None):
        newArray = np.zeros(shape).view(cls)
        newArray.data_type = data_type
        return newArray


    @property
    def data_type(self):
        return self._data_type

    @data_type.setter
    def data_type(self, value):

        if value is None:
            self._data_type = {}
            return

        self._data_type = value
        if self._data_type is not None:
            for d in self._data_type:
                if hasattr(self, d):
                    setattr(self, d, value[d])
                else:
                    raise excpt.InputDataError("data_type has "
                                                    + "unrecognised option '"
                                                    + str(d) + "'")

    @property
    def timebase(self):
        try:
            return self._timebase
        except AttributeError:
            return None
    
    @timebase.setter
    def timebase(self, value):
        self._check_data_type('timebase', value)
        self._timebase = value

    
    def _check_data_type(self, element, value):
        self._data_type[element] = value
        
    
    def _prep_reshape(self, n_sections = 1, use_time = None, use_turns = None):
        
        if use_turns is None and use_time is None:
            raise excpt.InputError("At least one of use_turns and "
                                        + "use_time should be defined")
            
        if use_time is not None:
            nPts = len(use_time)
        else:
            nPts = len(use_turns)
            
        if self.timebase == 'by_turn' and use_turns is None:
            raise excpt.InputError("If function is defined by_turn "
                                        + "use_turns must be given")

        if self.timebase == 'by_time' and use_time is None:
            raise excpt.InputError("If function is defined by_time "
                                        + "use_time must be given")

        return np.zeros([n_sections, nPts])
    
    
    def _comp_definition_reshape(self, n_sections, use_time, use_turns):

        if n_sections > 1 and self.shape[0] == 1:
            warnings.warn("multi-section required, but "
                          + str(self.__class__.__name__) + " function is single"
                          + " section, expanding to match n_sections")

        elif n_sections != self.shape[0]:
            raise excpt.DataDefinitionError("n_sections (" \
                                                 + str(n_sections) \
                                                 + ") does not match function "
                                                 + "definition ("\
                                                 + str(self.shape[0]) + ")")
    
        if self.timebase == 'by_time':
            if use_time is None:
                raise excpt.DataDefinitionError("Function is defined by "
                                                     + "time but use_time has"
                                                     + " not been given")
            if use_time[-1] > self[0, 0, -1]:
                warnings.warn("use_time extends beyond function definition, "
                              + "interpolation will assume fixed value")
            if use_time[0] < self[0, 0, 0]:
                warnings.warn("use_time starts before function definition, "
                              + "interpolation will assume fixed value")
        
        if self.timebase == 'by_turn':
            if use_turns is None:
                raise excpt.DataDefinitionError("Function is defined by "
                                                     + "turn but use_turns has"
                                                     + " not been given")
            if np.max(use_turns) > self.shape[1]:
                raise excpt.DataDefinitionError("Function does not have "
                                                     + "enough turns defined "
                                                     + "for maximum requested "
                                                     + "turn number")
    
    def _interpolate(self, section, use_time):

        if not np.all(np.diff(use_time) > 0):
            raise excpt.InputDataError("use_time is not monotonically "
                                            + "increasing")
            
        if self.interpolation == 'linear':
            return self._interpolate_linear(section, use_time)
        else:
            raise RuntimeError("Invalid interpolation requested")


    def _interpolate_linear(self, section, use_time):
        if self.shape[0] == 1:
            return np.interp(use_time, self[0, 0], self[0, 1])
        else:
            return np.interp(use_time, self[section, 0], self[section, 1])


    def reshape(self, n_sections = 1, use_time = None, use_turns = None):
        
        self._comp_definition_reshape(n_sections, use_time, use_turns)        
        newArray = self._prep_reshape(n_sections, use_time, use_turns)
        
        for s in range(n_sections):
            if self.timebase == 'single':

                if self.shape[0] == 1:
                    newArray[s] += self
                else:
                    newArray[s] += self[s]
    
            elif self.timebase == 'by_turn':

                if self.shape[0] == 1:
                    newArray[s] = self[0, use_turns]
                else:
                    newArray[s] = self[s, use_turns]
            
            elif self.timebase == 'by_time':
                    newArray[s] = self._interpolate(s, use_time)
        
        return newArray.view(self.__class__)


class _ring_function(_function):

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
        try:
            return self._sectioning
        except AttributeError:
            return None
    
    @sectioning.setter
    def sectioning(self, value):
        self._check_data_type('sectioning', value)
        self._sectioning = value


class _ring_program(_ring_function):

    conversions = {}

    def __new__(cls, *args, time = None, n_turns = None, interpolation = None):
        return super().__new__(cls, *args, time = time, n_turns = n_turns,
                               interpolation = interpolation)

    def to_momentum(self, inPlace = True, **kwargs):
        if self.source == 'momentum':
            return self._no_convert(inPlace)
        else:
            return self._convert('momentum', inPlace, **kwargs)

    def to_total_energy(self, inPlace = True, **kwargs):
        if self.source == 'energy':
            return self._no_convert(inPlace)
        else:
            return self._convert('energy', inPlace, **kwargs)

    def to_B_field(self, inPlace = True, **kwargs):
        if self.source == 'B_field':
            return self._no_convert(inPlace)
        else:
            return self._convert('B_field', inPlace, **kwargs)

    def to_kin_energy(self, inPlace = True, **kwargs):
        if self.source == 'kin_energy':
            return self._no_convert(inPlace)
        else:
            return self._convert('kin_energy', inPlace, **kwargs)   


    #TODO: Consider multi-section version
    @classmethod
    def combine_single_sections(cls, *args,
                                interpolation = None, **kwargs):

        if not all(isinstance(a, _ring_program) for a in args):
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
            raise exceptions.InputError(f"Available interpolation options are:\
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
        
            for s in range(self.shape[0]):
                if store_turns:
                    nTurns, useTurns, time, momentum = self._linear_interpolation(
                                                                mass,
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
        
        if inPlace:
            return self
        else:
            return super().__new__(self.__class__, *self)


    def _time_from_turn(self, mass, circumference):
        
        trev = rt.mom_to_trev(self[0], mass, circ=circumference)
        return np.cumsum(trev)


    def _linear_interpolation_no_turns(self, mass, circumference, time, 
                                       section):
        
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
        momentum_interp = np.asarray(momentum_interp)
        momentum_interp -= momentum_interp[0]
        momentum_interp /= momentum_interp[-1]
        momentum_interp *= input_momentum[-1] - input_momentum[0]
        momentum_interp += input_momentum[0]

        return nTurns, use_turns, time_interp, momentum_interp
        
        
    def _ramp_start_stop(self):
        
        if self.timebase != 'by_time':
            raise RuntimeError("Only implemented for by_time functions")
        
        time_start_ramp = np.max(self[0, 0][self[0, 1] == self[0, 1, 0]])
        time_end_ramp = np.max(self[0, 0][self[0, 1] == self[0, 1, -1]])

        return time_start_ramp, time_end_ramp

    @classmethod
    def _add_to_conversions(cls):
        cls.conversions[cls.source] = cls


class momentum_program(_ring_program):
    source = 'momentum'

class total_energy_program(_ring_program):
    source = 'energy'

class kinetic_energy_program(_ring_program):
    source = 'kin_energy'

class bending_field_program(_ring_program):
    source = 'B_field'

for data in [momentum_program, total_energy_program, kinetic_energy_program,
             bending_field_program]:
    data._add_to_conversions()


class momentum_compaction(_ring_function):
    
    def __new__(cls, *args, order = 0, time = None, n_turns = None, 
                interpolation = 'linear'):

        return super().__new__(cls, *args, time = time, 
                                 n_turns = n_turns, allow_single = True,
                                 interpolation = 'linear', order=order)

    @classmethod
    def combine_single_sections(cls, *args, interpolation = None):

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
        try:
            return self._order
        except AttributeError:
            return None
    
    @order.setter
    def order(self, value):
        self._check_data_type('order', value)
        self._order = value


class _RF_function(_function):
    
    def __new__(cls, *args, harmonics, time = None, n_turns = None, \
                interpolation = 'linear', allow_single = True, **kwargs):

        args = _expand_function(*args)

        _check_time_turns(time, n_turns)
        
        data_points, data_types = _get_dats_types(*args, time = time, \
                                                  n_turns = n_turns)
        
        _check_data_types(data_types, allow_single = allow_single)

        data_types, data_points = _expand_singletons(data_types, data_points)

        try:
            iter(harmonics)
        except TypeError:
            harmonics = (harmonics,)
        
        if len(data_points) != len(harmonics):
            raise excpt.InputError("Number of functions does not match " \
                                        + "number of harmonics")

        if not 'by_turn' in data_types:
            data_points = interpolate_input(data_points, data_types, 
                                            interpolation)
                
        data_type = {'timebase': data_types[0], 'harmonics': harmonics, 
                     **kwargs}
        
        return super().__new__(cls, data_points, data_type, 
                               interpolation)


    @property
    def harmonics(self):
        try:
            return self._harmonics
        except AttributeError:
            return None
    
    @harmonics.setter
    def harmonics(self, value):
        self._check_data_type('harmonics', value)
        self._harmonics = value


    #TODO: Safe treatment of use_turns > n_turns
    def reshape(self, harmonics = None, use_time = None, use_turns = None):
        
        if harmonics is None:
            harmonics = self.harmonics

        if use_turns is not None:
            use_turns = [int(turn) for turn in use_turns]

        newArray = self._prep_reshape(len(harmonics), 
                                      use_time = use_time, 
                                      use_turns = use_turns)
        
        for i, h in enumerate(harmonics):
            for j, s in enumerate(self.harmonics):
                if h == s:
                    break
            else:
                continue
            
            if self.timebase == 'single':
                newArray[i] += self[j]
    
            elif self.timebase == 'by_turn':
                newArray[i] = self[j, use_turns]
            
            elif self.timebase == 'by_time':
                newArray[i] = self._interpolate(j, use_time)
        
        newArray = newArray.view(self.__class__)

        newArray.data_type = {'timebase':  'interpolated',
                              'harmonics': harmonics}

        return newArray


class voltage_program(_RF_function):
    pass


class phase_program(_RF_function):
    pass


class _freq_phase_off(_RF_function):
    
    def calc_delta_omega(self, design_omega_rev):
        
        if not isinstance(self, phase_offset):
            raise RuntimeError("calc_delta_omega can only be used with a "
                                + "phase modulation function")
        
        if len(design_omega_rev.shape) == 2:
            if self.timebase not in ('by_time', 'interpolated'):
                raise RuntimeError("Time dependent design frequency requires" 
                                    + " time dependent offset frequency"
                                    + "or interpolated")

            if self.timebase == 'by_time':
                delta_phase = self.reshape(use_time = design_omega_rev[0])
            else:
                delta_phase = self.copy()

        else:
            if self.timebase not in ('by_turn', 'interpolated'):
                raise RuntimeError("Turn dependent design frequency requires" 
                                    + " turn dependent offset frequency" 
                                    + "or interpolated")
            delta_phase = self.copy()

        if delta_phase.shape[-1] != design_omega_rev.shape[-1]:
            raise excpt.InputError("design_omega_rev shape not correct")
        
        
        delta_omega = omega_offset.zeros(delta_phase.shape, self.data_type)
        delta_omega.timebase = 'interpolated'
        for i, h in enumerate(self.harmonics):
            if self.timebase in ('by_turn', 'interpolated'):
                delta_omega[i] = (design_omega_rev/(2*np.pi)) \
                                    * np.gradient(delta_phase[i])
            else:
                delta_omega[i] = (design_omega_rev[1]/(2*np.pi)) \
                                    * np.gradient(delta_phase[i])\
                                        /np.gradient(design_omega_rev[0])
        
        return delta_omega
    
    
    def calc_delta_phase(self, design_omega_rev, wrap=False):

        if not isinstance(self, omega_offset):
            raise RuntimeError("calc_delta_omega can only be used with a "
                               + "phase modulation function")
        
        if len(design_omega_rev.shape) == 2:
            if self.timebase not in ('by_time', 'interpolated'):
                raise RuntimeError("Time dependent design frequency requires" 
                                    + " time dependent offset frequency"
                                    + "or interpolated")

            if self.timebase == 'by_time':
                delta_omega = self.reshape(use_time = design_omega_rev[0])
            else:
                delta_omega = self.copy()

        else:
            if self.timebase not in ('by_turn', 'interpolated'):
                raise RuntimeError("Turn dependent design frequency requires" 
                                    + " turn dependent offset frequency" 
                                    + "or interpolated")
            delta_omega = self.copy()

        if delta_omega.shape[-1] != design_omega_rev.shape[-1]:
            raise excpt.InputError("design_omega_rev shape not correct")

        delta_phase = phase_offset.zeros(delta_omega.shape, self.data_type)
        delta_phase.timebase = 'interpolated'
        for i, h in enumerate(self.harmonics):
            if self.timebase in ('by_turn', 'interpolated'):
                delta_phase[i] = np.cumsum(2*np.pi\
                                           *(delta_omega[i]\
                                             /(h*design_omega_rev)))
            else:
                delta_phase[i] = np.cumsum(2*np.pi\
                                               *(delta_omega[i]\
                                                 /(h*design_omega_rev[1]))) \
                                    * np.gradient(design_omega_rev[0])
            
        return delta_phase


class phase_offset(_freq_phase_off):
    pass

class omega_offset(_freq_phase_off):
    pass


class _beam_data(_function):
    
    def __new__(cls, *args, units, time = None, n_turns = None, 
                interpolation = 'linear', **kwargs):
        
        _check_time_turns(time, n_turns)
            
        data_points, data_types = _get_dats_types(*args, time = time, \
                                                  n_turns = n_turns)
        
        _check_data_types(data_types, True)
        data_types, data_points = _expand_singletons(data_types, 
                                                     data_points)
        
        if 'by_turn' in data_types:
            _check_turn_numbers(data_points, data_types)
        else:
            data_points = interpolate_input(data_points, data_types, interpolation)

        if len(data_types) == 1:
            bunching = 'single_bunch'
        else:
            bunching = 'multi_bunch'
        
        data_type = {'timebase': data_types[0], 'bunching': bunching,
                     'units': units, **kwargs}
        
        return super().__new__(cls, data_points, data_type, interpolation)


    @property
    def bunching(self):
        try:
            return self._bunching
        except AttributeError:
            return None
    
    @bunching.setter
    def bunching(self, value):
        self._check_data_type('bunching', value)
        self._bunching = value
        
        
    @property
    def units(self):
        try:
            return self._units
        except AttributeError:
            return None
    
    @units.setter
    def units(self, value):
        self._check_data_type('units', value)
        self._units = value


class acceptance(_beam_data):
    
    def __new__(cls, *args, units = 'eVs', time = None, n_turns = None, 
                interpolation = 'linear'):
        
        return super().__new__(cls, *args, time = time, n_turns = n_turns,
                               interpolation = interpolation)


class emittance(_beam_data):
    
    def __new__(cls, *args, emittance_type = 'matched_area', units = 'eVs', 
                time = None, n_turns = None, interpolation = 'linear'):
        
        return super().__new__(cls, *args, 
                               emittance_type = emittance_type,
                               units = units, time = time, 
                               n_turns = n_turns,
                               interpolation = interpolation)


    @property
    def emittance_type(self):
        try:
            return self._emittance_type
        except AttributeError:
            return None
    
    @emittance_type.setter
    def emittance_type(self, value):
        self._check_data_type('emittance_type', value)
        self._emittance_type = value


class length(_beam_data):
    
    def __new__(cls, *args, length_type = 'full_length', units = 's',
                time = None, n_turns = None, interpolation = 'linear'):
        
        return super().__new__(cls, *args, length_type = length_type, 
                               units = units, time = time, 
                               n_turns = n_turns,
                               interpolation = interpolation)


    @property
    def length_type(self):
        try:
            return self._length_type
        except AttributeError:
            return None
    
    @length_type.setter
    def length_type(self, value):
        self._check_data_type('length_type', value)
        self._length_type = value
        

class height(_beam_data):
    
    def __new__(cls, *args, height_type = 'half_height', units = 'eV',
                time = None, n_turns = None, interpolation = 'linear'):
        
        return super().__new__(cls, *args, height_type = height_type, 
                               units = units, time = time, n_turns = n_turns,
                               interpolation = interpolation)


    @property
    def height_type(self):
        try:
            return self._height_type
        except AttributeError:
            return None
    
    @height_type.setter
    def height_type(self, value):
        self._check_data_type('height_type', value)
        self._height_type = value


class synchronous_phase(_beam_data):
    
    def __new__(cls, *args, units = 's', time = None, n_turns = None, 
                interpolation = 'linear'):
        
        return super().__new__(cls, *args, units = units, time = time, 
                               n_turns = n_turns, 
                               interpolation = interpolation)



###############################################
####FUNCTIONS TO HELP IN DATA TYPE CREATION####
###############################################
    
def _expand_singletons(data_types, data_points):
    
        if 'by_turn' in data_types:
            n_turns = _check_turn_numbers(data_points, data_types, \
                                          allow_single=True)

            if 'single' in data_types:
                for i, t in enumerate(data_types):
                    if t == 'single':
                        data_types[i] = 'by_turn'
                        data_points[i] = [data_points[i]]*n_turns

        if 'by_time' in data_types and 'single' in data_types:

            for i, t in enumerate(data_types):
                if t == 'by_time':
                    useTime = data_points[i][0]
                    break

            for i, t in enumerate(data_types):
                if t == 'single':
                    data_types[i] = 'by_time'
                    data_points[i] = np.array([useTime, \
                                               [data_points[i]]*len(useTime)])
    
        return data_types, data_points

#For functions defined by turn number, check all have same number of turns
def _check_turn_numbers(data_points, data_types, allow_single=False):

    lengths = []
    for datPt, datType in zip(data_points, data_types):
        if datType == 'by_turn':
            lengths.append(len(datPt))

    if not allow_single:
        if len(lengths) != len(data_types):
            raise excpt.InputError("Functions with single and by_turn " \
                                        + "together not allowed")
            
    if not all(length == lengths[0] for length in lengths):
        raise excpt.DataDefinitionError("Functions defined by " \
                                            + "turn with unequal " \
                                            + "numbers of turns")
    
    return lengths[0]


def _check_data_types(data_types, allow_single = False):

    comparator = []
    for t in data_types:
        if t != 'single':
            comparator.append(t)
            break
    
    if len(comparator) == 0:
        comparator.append('single')

    if allow_single and 'single' not in comparator:
        comparator.append('single')

    gen = (datType in comparator for datType in data_types)

    if not all(gen):
        raise excpt.DataDefinitionError("Input programs " \
                                     + "follow different conventions")
            
#Raise excpt if both time and n_turns are not None
def _check_time_turns(time, n_turns):
    if time is not None and n_turns is not None:
            raise excpt.InputError("time and n_turns cannot both be "
                                        + "specified")

#Loop over _check_dims for all *args and return corresponding data_points 
#and data_types
def _get_dats_types(*args, time, n_turns):
    
    data_points = []
    data_types = []
    
    for arg in args:
        data_point, data_type = _check_dims(arg, time, n_turns)
        data_points.append(data_point)
        data_types.append(data_type)
    
    return data_points, data_types


#Identify if data is single valued, by_turn, or by_time
def _check_dims(data, time = None, n_turns = None):

    #Check and handle single valued data
    #if not single valued coerce to numpy array and continue
    try:
        iter(data)
        data = np.array(data)
    except TypeError:
        if n_turns is None:
            return data, 'single'
        else:
            return [data]*n_turns, 'by_turn'

    #If n_turns specified and data is not single valued it should be 
    #of len(n_turns)
    if n_turns is not None:
        if len(data) == n_turns:
            return data, 'by_turn'
        else:
            raise excpt.InputError("Input length does not match n_turns")
    
    elif time is not None:
        if data.shape[0] == 2 and hasattr(data[0], '__iter__'):
            raise excpt.InputError("Data has been passed with " \
                                        + "[time, value] format and time " \
                                        + "defined, only 1 should be given")
        else:
            #If time is passed don't return, use test below avoids duplication
            if len(data) == len(time):
                data = np.array([time, data])
            else:
                raise excpt.InputError("time and data are of unequal" \
                                            + " length")

    #If data has shape (2, n) data[0] is taken as time
    if data.shape[0] == 2 and len(data.shape) == 2:
        return data, 'by_time'
    #if data has shape (n,) data[0] is taken as momentum by turn
    elif len(data.shape) == 1:
        return data, 'by_turn'

    raise excpt.InputError("Input data not understood")



def interpolate_input(data_points, data_types, interpolation = 'linear'):
    
    if interpolation != 'linear':
        raise RuntimeError("Only linear interpolation defined")
    
    if all(t == 'single' for t in data_types):
        return data_points
    
    if data_types[0] != 'by_time':
        excpt.DataDefinitionError("Interpolation only possible if functions "
                                       + "are defined by time")
    
    input_times = []
    for d in data_points:
        input_times += d[0].tolist()
    
    interp_times = sorted(set(input_times))

    for i in range(len(data_points)):
         interp_data = np.interp(interp_times, data_points[i][0], \
                                 data_points[i][1])
         data_points[i] = np.array([interp_times, interp_data])

    return data_points


def _from_function(inputArray, targetClass, **kwargs):
    
    if isinstance(inputArray, targetClass):
        return inputArray
    
    else:
        return None


def _expand_function(*args):
    newArgs = []
    for a in args:
        if isinstance(a, _function) or isinstance(a, bf.machine_program):
            newArgs.append(*a)
        else:
            newArgs.append(a)
    return tuple(newArgs)

############################################
####LOCAL EQUIVALENTS TO NUMPY FUNCTIONS####
############################################
    
def stack(*args, interpolation = 'linear'):

    if not all(hasattr(a, '__iter__') for a in args):
        raise exceptions.InputError("All args should be iterable, either as a "
                                   + "datatype object, or a list or tuple with"
                                   + " start and/or end times for the stack")

    start_time = []
    functions = []
    stop_time = []
    for a in args:
        
        if isinstance(a, _function):
            start_time.append(None)
            functions.append(a)
            stop_time.append(None)
            continue

        elif len(a) == 3:
            start_time.append(a[0])
            functions.append(a[1])
            stop_time.append(a[2])

        elif len(a) == 2:
            
            if isinstance(a[0], _function):
                start_time.append(None)
                functions.append(a[0])
                stop_time.append(a[1])

            elif isinstance(a[1], _function):
                start_time.append(a[0])
                functions.append(a[1])
                stop_time.append(None)
            else:
                raise exceptions.InputError("All elements must have a "
                                       + "datatype included")
        
        elif len(a) == 1:
            start_time.append(None)
            functions.append(a[0])
            stop_time.append(None)

        else:
            raise exceptions.InputError("If passing an iterable it must have "
                                   + "a maximum length of 3")

    if not all(isinstance(f, _function) for f in functions):
        raise exceptions.InputError("All functions should be datatype objects")

    timebases = [f.timebase for f in functions]

    if 'single' in timebases:
        raise exceptions.InputError("Single valued functions cannot be stacked")

    if not all(t == timebases[0] for t in timebases):
        raise exceptions.InputError("Only functions with the same timebase "
                                    +"can be stacked")

    if not all(type(f) == type(functions[0]) for f in functions):
        raise exceptions.InputError("All functions should be the same type")

    if not all(f.data_type == functions[0].data_type for f in functions):
        raise exceptions.InputError("All function should have the same "
                                    + "data_type")

    nSections = functions[0].shape[0]

    if timebases[0] == 'by_time':
        useTimes = []
        subFunctions = [[] for n in range(nSections)]
        
        for start, f, stop in zip(start_time, functions, stop_time):

            if start is None:
                start = f[0, 0, 0]
            if stop is None:
                stop = f[0, 0, -1]

            usePts = np.where((f[0, 0] > start) * (f[0, 0] < stop))[0]
            if start == stop:
                interp_time = [start]
            else:
                interp_time = [start] + f[0, 0, usePts].tolist() + [stop]
            
            useTimes += interp_time
            
            reshaped = f.reshape(use_time = interp_time)
        
            for n in range(nSections):
                subFunctions[n] += reshaped[n].tolist()
        
        newArray = functions[0].__class__.zeros([nSections, 2, len(useTimes)])
        for i, f in enumerate(subFunctions):
            newArray[i] = [useTimes, f]
        
        newArray.data_type = functions[0].data_type
        newArray.interpolation = interpolation
        
        return newArray
    
    for f in subFunctions:
        print(f)
        plt.plot(useTimes, f)
    plt.show()
    
#    for u, f in zip(useTimes, subFunctions):
#        plt.plot(u, f[0])
#    plt.show()