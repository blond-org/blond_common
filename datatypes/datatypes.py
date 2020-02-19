#General imports
import numpy as np
import sys
import os

#Common imports
from ..devtools import exceptions


class _function(np.ndarray):
    
    def __new__(cls, input_array, data_type=None):
        
        if data_type is None:
            raise exceptions.InputError("data_type must be specified")
        
        try:
            obj = np.asarray(input_array).view(cls)
        except ValueError:
            raise exceptions.InputError("Function components could not be " \
                                        + "correctly coerced into ndarray, " \
                                        + "check input dimensionality")

        obj.data_type = data_type
        
        obj.func_type = data_type[0]
        obj.time_base = data_type[1]
        obj.sectioning = data_type[2]        
        
        return obj
    
    def __array_finalize__(self, obj):
        
        if obj is None:
            return
        
        self.data_type = getattr(obj, 'data_type', None)

        if self.data_type is not None:
            self.func_type = self.data_type[0]
            self.time_base = self.data_type[1]
            self.sectioning = self.data_type[2] 
    
    
    def reshape(self, n_sections = 1, use_time = None, use_turns = None):
        
        if use_turns is None and use_time is None:
            raise exceptions.InputError("At least one of use_turns and "
                                        + "use_time should be defined")
            
        if use_time is not None:
            nPts = len(use_time)
        else:
            nPts = len(use_turns)

        newArray = np.zeros([n_sections, nPts])

        for s in range(n_sections):        
            if self.time_base == 'single':
                if self.sectioning == 'single_section':
                    newArray[s] += self
    
            elif self.time_base == 'by_turn':
                if self.sectioning == 'single_section':
                    newArray[s] = self[0, use_turns]
            
            elif self.time_base == 'by_time':
                if self.sectioning == 'single_section':
                    pass
            
        
        return newArray.view(self.__class__)
        
        


class _ring_function(_function):

    def __new__(cls, *args, func_type, time = None, n_turns = None):
        
        _check_time_turns(time, n_turns)
            
        data_points, data_types = _get_dats_types(*args, time = time, \
                                                  n_turns = n_turns)
        _check_data_types(data_types)

        if 'by_turn' in data_types:
            _check_turn_numbers(data_points, data_types)
            
        if len(data_types) == 1:
            return super().__new__(cls, data_points, \
                        (func_type, data_types[0], 'single_section'))
        else:
            return super().__new__(cls, data_points, \
                        (func_type, data_types[0], 'multi_section'))


class RF_section_function(_function):
    
    def __new__(cls, *args, harmonics, time = None, n_turns = None, \
                interpolation = None):
        
        _check_time_turns(time, n_turns)
        
        data_points, data_types = _get_dats_types(*args, time = time, \
                                                  n_turns = n_turns)
        
        _check_data_types(data_types, allow_single=True)

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

        try:
            iter(harmonics)
        except TypeError:
            harmonics = (harmonics,)
        
        if len(data_points) != len(harmonics):
            raise exceptions.InputError("Number of functions does not match " \
                                        + "number of harmonics")
                
        if interpolation is None or len(data_points) == 1:
            return super().__new__(cls, data_points, \
                                    ('RF', data_types[0], harmonics))
        
        if interpolation is not None and data_types[0] != 'by_time':
            raise exceptions.DataDefinitionError("Interpolation only possible" \
                                                 + " if functions are defined" \
                                                 + " by time")

        if interpolation != 'linear':
            raise RuntimeError("Only linear interpolation currently available")
            
        input_times = []
        for d in data_points:
            input_times += d[0].tolist()
        
        interp_times = sorted(set(input_times))

        for i in range(len(data_points)):
             interp_data = np.interp(interp_times, data_points[i][0], \
                                     data_points[i][1])
             data_points[i] = np.array([interp_times, interp_data])
        
        return super().__new__(cls, data_points, ('RF', data_types[0], \
                                                  harmonics))



class ring_program(_ring_function):
    
    def __new__(cls, *args, data_type='momentum', time = None, n_turns = None):
        return super().__new__(cls, *args, func_type = data_type, time = time, 
                         n_turns = n_turns)


class momentum_compaction(_ring_function):
    
    def __new__(cls, *args, order = 0, time = None, n_turns = None):
        return super().__new__(cls, *args, func_type = order, time = time, 
                         n_turns = n_turns)


#For functions defined by turn number, check all have same number of turns
def _check_turn_numbers(data_points, data_types, allow_single=False):

    lengths = []
    for datPt, datType in zip(data_points, data_types):
        if datType == 'by_turn':
            lengths.append(len(datPt))

    if not allow_single:
        if len(lengths) != len(data_types):
            raise exceptions.InputError("Functions with single and by_turn " \
                                        + "together not allowed")
            
    if not all(length == lengths[0] for length in lengths):
        raise exceptions.DataDefinitionError("Functions defined by " \
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
        raise exceptions.DataDefinitionError("Input programs " \
                                     + "follow different conventions")
            
#Raise exceptions if both time and n_turns are not None
def _check_time_turns(time, n_turns):
    if time is not None and n_turns is not None:
            raise exceptions.InputError("time and n_turns cannot both be specified")

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

    #If n_turns specified and data is not single valued it should be of len(n_turns)
    if n_turns is not None:
        if len(data) == n_turns:
            return data, 'by_turn'
        else:
            raise exceptions.InputError("Input length does not match n_turns")
    
    elif time is not None:
        if data.shape[0] == 2:
            raise exceptions.InputError("Data has been passed with " \
                                        + "[time, value] format and time " \
                                        + "defined, only 1 should be given")
        else:
            #If time is passed don't return, use test below avoids duplication
            if len(data) == len(time):
                data = np.array([time, data])
            else:
                raise exceptions.InputError("time and data are of unequal" \
                                            + " length")

    #If data has shape (2, n) data[0] is taken as time
    if data.shape[0] == 2 and len(data.shape) == 2:
        return data, 'by_time'
    #if data has shape (n,) data[0] is taken as momentum by turn
    elif len(data.shape) == 1:
        return data, 'by_turn'

    raise exceptions.InputError("Input data not understood")