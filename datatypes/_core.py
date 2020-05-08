# General imports
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
    r"""
    Base class defining the functionality common to all datatypes objects.  
    Inherits from np.ndarray.
    
    Parameters
    ----------
    input_array : float, list, list of lists
        Data to be cast as a numpy array
    data_type : dict
        Dictionary of relevant information to define the content of the 
        datatype object, always includes 'timebase' key as well as those
        needed by different subclasses
    interpolation : str
        Identifier of the type of interpolation to be used when reshaping
        the array, currently only 'linear' has been implemented
    
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
    """
    
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
        """
        

        Parameters
        ----------
        obj : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if obj is None:
            return

        self.data_type = getattr(obj, 'data_type', None)


    @classmethod
    def zeros(cls, shape, data_type = None):
        """
        

        Parameters
        ----------
        cls : TYPE
            DESCRIPTION.
        shape : TYPE
            DESCRIPTION.
        data_type : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        newArray : TYPE
            DESCRIPTION.

        """
        newArray = np.zeros(shape).view(cls)
        newArray.data_type = data_type
        return newArray


    @property
    def data_type(self):
        """
        Get or set the data_type.  Setting the data_type will update all
        attributes of the object.
        """
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
        """
        Get or set the timebase.  Setting the timebase will update the
        data_type dict.
        """
        try:
            return self._timebase
        except AttributeError:
            return None
    
    @timebase.setter
    def timebase(self, value):
        self._check_data_type('timebase', value)
        self._timebase = value

    
    def _check_data_type(self, element, value):
        """
        

        Parameters
        ----------
        element : TYPE
            DESCRIPTION.
        value : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self._data_type[element] = value
        
    
    def _prep_reshape(self, n_sections = 1, use_time = None, use_turns = None):
        """
        

        Parameters
        ----------
        n_sections : TYPE, optional
            DESCRIPTION. The default is 1.
        use_time : TYPE, optional
            DESCRIPTION. The default is None.
        use_turns : TYPE, optional
            DESCRIPTION. The default is None.

        Raises
        ------
        excpt
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
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
        """
        

        Parameters
        ----------
        n_sections : TYPE
            DESCRIPTION.
        use_time : TYPE
            DESCRIPTION.
        use_turns : TYPE
            DESCRIPTION.

        Raises
        ------
        excpt
            DESCRIPTION.

        Returns
        -------
        None.

        """
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
            if np.max(use_turns) >= self.shape[1]:
                raise excpt.DataDefinitionError("Function does not have "
                                                     + "enough turns defined "
                                                     + "for maximum requested "
                                                     + "turn number")
    
    def _interpolate(self, section, use_time):
        """
        

        Parameters
        ----------
        section : TYPE
            DESCRIPTION.
        use_time : TYPE
            DESCRIPTION.

        Raises
        ------
        excpt
            DESCRIPTION.
        RuntimeError
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if not np.all(np.diff(use_time) > 0):
            raise excpt.InputDataError("use_time is not monotonically "
                                            + "increasing")
            
        if self.interpolation == 'linear':
            return self._interpolate_linear(section, use_time)
        else:
            raise RuntimeError("Invalid interpolation requested")


    def _interpolate_linear(self, section, use_time):
        """
        

        Parameters
        ----------
        section : TYPE
            DESCRIPTION.
        use_time : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if self.shape[0] == 1:
            return np.interp(use_time, self[0, 0], self[0, 1])
        else:
            return np.interp(use_time, self[section, 0], self[section, 1])


    def reshape(self, n_sections = 1, use_time = None, use_turns = None):
        """
        

        Parameters
        ----------
        n_sections : TYPE, optional
            DESCRIPTION. The default is 1.
        use_time : TYPE, optional
            DESCRIPTION. The default is None.
        use_turns : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
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


###############################################
####FUNCTIONS TO HELP IN DATA TYPE CREATION####
###############################################
    
def _expand_singletons(data_types, data_points):
    """
    

    Parameters
    ----------
    data_types : TYPE
        DESCRIPTION.
    data_points : TYPE
        DESCRIPTION.

    Returns
    -------
    data_types : TYPE
        DESCRIPTION.
    data_points : TYPE
        DESCRIPTION.

    """
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
    """
    

    Parameters
    ----------
    data_points : TYPE
        DESCRIPTION.
    data_types : TYPE
        DESCRIPTION.
    allow_single : TYPE, optional
        DESCRIPTION. The default is False.

    Raises
    ------
    excpt
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
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
    """
    

    Parameters
    ----------
    data_types : TYPE
        DESCRIPTION.
    allow_single : TYPE, optional
        DESCRIPTION. The default is False.

    Raises
    ------
    excpt
        DESCRIPTION.

    Returns
    -------
    None.

    """
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
    """
    

    Parameters
    ----------
    time : TYPE
        DESCRIPTION.
    n_turns : TYPE
        DESCRIPTION.

    Raises
    ------
    excpt
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if time is not None and n_turns is not None:
            raise excpt.InputError("time and n_turns cannot both be "
                                        + "specified")

#Loop over _check_dims for all *args and return corresponding data_points 
#and data_types
def _get_dats_types(*args, time, n_turns):
    """
    

    Parameters
    ----------
    *args : TYPE
        DESCRIPTION.
    time : TYPE
        DESCRIPTION.
    n_turns : TYPE
        DESCRIPTION.

    Returns
    -------
    data_points : TYPE
        DESCRIPTION.
    data_types : TYPE
        DESCRIPTION.

    """
    data_points = []
    data_types = []
    
    for arg in args:
        data_point, data_type = _check_dims(arg, time, n_turns)
        data_points.append(data_point)
        data_types.append(data_type)
    
    return data_points, data_types


#Identify if data is single valued, by_turn, or by_time
def _check_dims(data, time = None, n_turns = None):
    """
    

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    time : TYPE, optional
        DESCRIPTION. The default is None.
    n_turns : TYPE, optional
        DESCRIPTION. The default is None.

    Raises
    ------
    excpt
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    str
        DESCRIPTION.

    """
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



def _interpolate_input(data_points, data_types, interpolation = 'linear'):
    """
    

    Parameters
    ----------
    data_points : TYPE
        DESCRIPTION.
    data_types : TYPE
        DESCRIPTION.
    interpolation : TYPE, optional
        DESCRIPTION. The default is 'linear'.

    Raises
    ------
    RuntimeError
        DESCRIPTION.

    Returns
    -------
    data_points : TYPE
        DESCRIPTION.

    """
    if interpolation != 'linear':
        raise RuntimeError("Only linear interpolation defined")
    
    if all(t == 'single' for t in data_types):
        return data_points
    """
    Class defining the length of the beam
    Parameters
    ----------
    *args : float, 1D iterable of floats, 2D iterable of floats
    length_type : str
    units : str
    time : iterable of floats
    n_turns : int
    interpolation : str
    
    Attributes
    ----------
    As _function plus
    units : str
    bunching : str
    length_type : str
    """
    
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
    """
    

    Parameters
    ----------
    inputArray : TYPE
        DESCRIPTION.
    targetClass : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    inputArray : TYPE
        DESCRIPTION.

    """
    if isinstance(inputArray, targetClass):
        return inputArray
    
    else:
        return None


def _expand_function(*args):
    """
    

    Parameters
    ----------
    *args : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
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
    """
    

    Parameters
    ----------
    *args : TYPE
        DESCRIPTION.
    interpolation : TYPE, optional
        DESCRIPTION. The default is 'linear'.

    Raises
    ------
    excpt
        DESCRIPTION.

    Returns
    -------
    newArray : TYPE
        DESCRIPTION.

    """
    if not all(hasattr(a, '__iter__') for a in args):
        raise excpt.InputError("All args should be iterable, either as a "
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
                raise excpt.InputError("All elements must have a "
                                       + "datatype included")
        
        elif len(a) == 1:
            start_time.append(None)
            functions.append(a[0])
            stop_time.append(None)

        else:
            raise excpt.InputError("If passing an iterable it must have "
                                   + "a maximum length of 3")

    if not all(isinstance(f, _function) for f in functions):
        raise excpt.InputError("All functions should be datatype objects")

    timebases = [f.timebase for f in functions]

    if 'single' in timebases:
        raise excpt.InputError("Single valued functions cannot be stacked")

    if not all(t == timebases[0] for t in timebases):
        raise excpt.InputError("Only functions with the same timebase "
                                    +"can be stacked")

    if not all(type(f) == type(functions[0]) for f in functions):
        raise excpt.InputError("All functions should be the same type")

    if not all(f.data_type == functions[0].data_type for f in functions):
        raise excpt.InputError("All function should have the same "
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