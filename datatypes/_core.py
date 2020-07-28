# General imports
import numpy as np
import numbers
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
    
    def __new__(cls, input_array, data_type, interpolation = None,
                dtype = None):

        try:
            obj = np.asarray(input_array, dtype=dtype).view(cls)
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
        obj : datatype
        """
        if obj is None:
            return

        #Duplicate data_type dict to prevent two objects referencing the same
        # dict
        try:
            self.data_type = {**getattr(obj, 'data_type')}
        except AttributeError:
            self.data_type = None


    def __add__(self, other):
        return self._operate(other, np.add, inPlace = False)

    def __iadd__(self, other):
        self._operate(other, np.add, inPlace = True)
        return self

    __radd__ = __add__
    __riadd__ = __iadd__


    def __sub__(self, other):
        return self._operate(other, np.subtract, inPlace = False)

    def __isub__(self, other):
        self._operate(other, np.subtract, inPlace = True)
        return self

    def __mul__(self, other):
        return self._operate(other, np.multiply, inPlace = False)

    def __imul__(self, other):
        self._operate(other, np.multiply, inPlace = True)
        return self

    __rmul__ = __mul__
    __rimul__ = __imul__

    def __truediv__(self, other):
        return self._operate(other, np.true_divide, inPlace = False)

    def __itruediv__(self, other):
        self._operate(other, np.true_divide, inPlace = True)
        return self

    def __ge__(self, other):
        #Casting to regular numpy array as _function of bool makes no sense.
        return self._operate(other, np.greater_equal,
                             inPlace = False).view(np.ndarray)

    def __gt__(self, other):
        #Casting to regular numpy array as _function of bool makes no sense.
        return self._operate(other, np.greater,
                             inPlace = False).view(np.ndarray)

    def __le__(self, other):
        #Casting to regular numpy array as _function of bool makes no sense.
        return self._operate(other, np.less_equal,
                             inPlace = False).view(np.ndarray)

    def __lt__(self, other):
        #Casting to regular numpy array as _function of bool makes no sense.
        return self._operate(other, np.less,
                             inPlace = False).view(np.ndarray)


    def _operate(self, other, operation, inPlace = False):
        if isinstance(other, self.__class__):
            self._check_data_and_type(other)
            newArray = self._operate_equivalent_functions(other, operation)
        else:
            newArray = self._operate_other_functions(other, operation)

        if not inPlace:
            return newArray
        if inPlace:
            if self.timebase == 'by_time':
                self[:,1,:] = newArray[:,1,:]
            elif self.timebase == 'single':
                self[()] = newArray[()]
            else:
                self[:] = newArray[:]

        return


    def _operate_equivalent_functions(self, other, operation):

        if self.timebase != other.timebase:
            #should never be reached
            raise TypeError("Only functions with the same timebase can be "
                            + "used.")

        if self.timebase == 'by_time':
            return self._operate_general(other[:,1,:], operation)
        else:
            return self._operate_general(other, operation)


    def _operate_other_functions(self, other, operation):
        if isinstance(other, numbers.Number):
            return self._operate_general(other, operation)
        else:
            try:
                otherShape = other.shape
            except AttributeError:
                raise RuntimeError("unrecognised other")
            else:
                if otherShape == self.shape:
                    return self._operate_general(other, operation)
                else:
                    raise RuntimeError("other shape incorrect")


    def _operate_general(self, other, operation):

        newArray = self.copy()
        if self.timebase == 'by_time':
            if len(self.shape) == 3:
                newArray[:,1,:] = operation(self[:,1,:], other)
            else:
                #TODO: Is this safe?
                newArray[:] = operation(self, other)
        elif self.timebase == 'single':
            newArray[()] = operation(self, other)
        else:
            if self.shape == ():
                newArray[()] = operation(self, other)
            else:
                newArray[:] = operation(self, other)
        return newArray


    def _type_check(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError("Datatypes should be the same, but they are "
                            + f"{self.__class__} and {other.__class__}.")


    def _data_check(self, other):
        if self.data_type != other.data_type:
            raise TypeError("Datatypes should have the same `data_type` dict "
                            +f"but they are {self.data_type} and "
                            +f"{other.data_type}")


    def _check_data_and_type(self, other):
        self._type_check(other)
        self._data_check(other)


    @classmethod
    def zeros(cls, shape, data_type = None):
        """
        Create an empty array of given shape, with required data_type dict.

        Parameters
        ----------
        shape : iterable 
            The shape of the new array
        data_type : dict, optional
            The dict defining the data_type attribute of the new array.
            The default is None.

        Returns
        -------
        newArray : datatype
            The new datatype array
        """
        newArray = np.zeros(shape).view(cls)
        newArray.data_type = data_type
        return newArray


    @property
    def data_type(self):
        """
        Get or set the data_type.  Setting the data_type will update all
        attributes of the object identified in the dict.
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
                    raise excpt.InputDataError("data_type has unrecognised "
                                               + f"option '{d}'")
        return

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


    @property
    def interpolation(self):
        """
        Get or set the timebase.  Setting the timebase will update the
        data_type dict.
        """
        try:
            return self._interpolation
        except AttributeError:
            return None

    @interpolation.setter
    def interpolation(self, value):
        self._check_data_type('interpolation', value)
        self._interpolation = value


    def _check_data_type(self, element, value):
        """
        Attempt to assign a value to the data_type dict

        Parameters
        ----------
        element : str
            str identifying the element of the data_type dict to be updated.
        value : any
            The new element to be added to the data_type dict.
        """
        self._data_type[element] = value


    def _prep_reshape(self, n_sections = 1, use_time = None, use_turns = None,
                      store_time = False):
        """
        Construct an empty numpy array of the required shape for the reshaped
        array to populate

        Parameters
        ----------
        n_sections : int, optional
            The number of sections described by the array. The default is 1.
        use_time : iterable of float, optional
            The times that the array will be interpolated onto.
            The default is None.
        use_turns : iterable of int, optional
            The turn numberss to be extracted from the array. 
            The default is None.

        Raises
        ------
        excpt.InputError
            If neither use_turns nor use_time is defined an exception is
            raised.

        Returns
        -------
        np.ndarray
            The new array to be populated when reshaping the data.
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

        if store_time:
            return self.zeros([n_sections, 2, nPts],
                                      data_type = {**self.data_type})
        else:
            return self.zeros([n_sections, nPts],
                              data_type = {**self.data_type})


    def _comp_definition_reshape(self, n_sections, use_time, use_turns):
        """
        Compare the required reshaping data with the contents of the datatype
        to confirm it is suitable.

        Parameters
        ----------
        n_sections : int
            Number of sections required for the new array.
        use_time : iterable of floats
            Times to be used for the interpolation.
        use_turns : iterable of ints
            Turn numbers to be used for new array.

        Raises
        ------
        excpt.DataDefinitionError
            If a function self has multiple sections and n_sections does not
            match the number of sections a DataDefintionError is raised.
            If a function is defined by_time (by_turn) and the reshaping 
            requires by_turn (by_time) a DataDefinitionError is raised.
            If the funtion has fewer turns of data available than required
            for the use_turns bassed, a DataDefinitionError is raised.
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
        Control the interpolation of the data

        Parameters
        ----------
        section : int
            Section number to be interpolated.
        use_time : iterable of floats
            The times the array will be interpolated onto.

        Raises
        ------
        excpt.InputDataError
            If use_time is not monotonically increase an InputDataError is
            raised.
        RuntimeError
            At present only linear interpolation is available, if another
            type is requested a RuntimeError is raised.

        Returns
        -------
        np.ndarray
            The interpolated array.

        """
        if not np.all(np.diff(use_time) > 0):
            raise excpt.InputDataError("use_time is not monotonically "
                                            + "increasing")
            
        if self.interpolation == 'linear':
            return self._interpolate_linear(section, use_time)
        else:
            raise NotImplementedError("Only linear interpolation implemented, "
                                      +f"{self.interpolation} not available.")


    def _interpolate_linear(self, section, use_time):
        """
        Make a linear interpolation

        Parameters
        ----------
        section : int
            Section number to be interpolated.
        use_time : iterable of floats
            The times the array will be interpolated onto.

        Returns
        -------
        np.ndarray
            The newly interpolated array.
        """
        if self.shape[0] == 1:
            return np.interp(use_time, self[0, 0], self[0, 1])
        else:
            return np.interp(use_time, self[section, 0], self[section, 1])


    def reshape(self, n_sections = 1, use_time = None, use_turns = None,
                store_time = False):
        """
        Reshape the datatype array to the given number of sections and either
        the given use_time or given use_turns.

        Parameters
        ----------
        n_sections : int, optional
            The number of sections required for the new array.
            The default is 1.
        use_time : iterable of floats, optional
            The times that the array will be interpolated on to.
            The default is None.
        use_turns : iterable of ints, optional
            The turn numbers to be used for the new array. The default is None.

        Returns
        -------
        datatype
            The newly interpolated array.
        """

        if self.timebase == 'by_turn' and store_time:
            raise excpt.InputError("A function defined by_turn cannot have "
                                   + "store_time=True")

        self._comp_definition_reshape(n_sections, use_time, use_turns)
        interpArray = self._prep_reshape(n_sections, use_time, use_turns,
                                         store_time)

        for s in range(n_sections):
            if self.timebase == 'single':
                if self.shape[0] == 1:
                    interpArray[s] += self
                else:
                    interpArray[s] += self[s]

            elif self.timebase == 'by_turn':
                if self.shape[0] == 1:
                    interpArray[s] = self[0, use_turns]
                else:
                    interpArray[s] = self[s, use_turns]

            elif self.timebase == 'by_time':
                    interpArray[s] = self._interpolate(s, use_time)

        if store_time:
            interpArray[:,0,:] = use_time
            interpArray.timebase = 'by_time'
        else:
            interpArray.timebase = 'interpolated'

        return interpArray


###############################################
####FUNCTIONS TO HELP IN DATA TYPE CREATION####
###############################################

def _expand_singletons(data_types, data_points):
    """
    Function to expand single points of data to the required shape for the
    new array.

    Parameters
    ----------
    data_types : list of str
        The preliminary data_type for each data_point.
    data_points : list of floats and/or iterables
        The data to be checked for compatability.

    Returns
    -------
    data_types : list of str
        The updated data_types.
    data_points : 
        The updated data_points.
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


def _check_turn_numbers(data_points, data_types, allow_single=False):
    """
    Function to check that all given data covers the same number of turns

    Parameters
    ----------
    data_points : iterable
        The data_points to be used in the datatype array.
    data_types : iterable of str
        The timebase for each member of data_points.
    allow_single : bool, optional
        Identify if single values are allowed. The default is False.

    Raises
    ------
    excpt.InputError
        If allow_single is False and both by_turn and single valued data is
        given an InputError is raised.
    except.DataDefinitionError
        If the given arrays are not of equal length a DataDefintionError is
        raised.

    Returns
    -------
    int
        The number of turns of data.
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
    Function to check that all data has the same type, allows singles if 
    flagged

    Parameters
    ----------
    data_types : iterable of str
        The preliminary timebases for the data.
    allow_single : bool, optional
        Define if single valued data can be mixed with time dependent data.
        The default is False.

    Raises
    ------
    excpt.DataDefinitionError
        If the input data is a mix of timebase conventions a
        DataDefinitionError is raised.
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


#TODO: Where used replace with SingleNotNone assertion
def _check_time_turns(time, n_turns):
    """
    Function to check if time and n_turns are both defined

    Parameters
    ----------
    time : iterable of float
        The time for the data.
    n_turns : int
        The number of turns for the data.

    Raises
    ------
    excpt.InputError
        Raises an InputError if both time and n_turns are not None.
    """
    if time is not None and n_turns is not None:
            raise excpt.InputError("time and n_turns cannot both be "
                                        + "specified")


#Loop over _check_dims for all *args and return corresponding data_points 
#and data_types
def _get_dats_types(*args, time, n_turns):
    """
    Function to take the given arguments and compare their dimensions
    with the given time and n_turns

    Parameters
    ----------
    *args : iterable of float and/or iterables
        The data to be used to construct the new array.
    time : iterable of float
        The times used for the new array.
    n_turns : int
        The number of turns of data.

    Returns
    -------
    data_points : iterable of floats and/or iterables
        The data to be used for the new array.
    data_types : list
        The preliminary timebases for the data.
    """
    data_points = []
    data_types = []
    
    for arg in args:
        data_point, data_type = _check_dims(arg, time, n_turns)
        data_points.append(data_point)
        data_types.append(data_type)
    
    return data_points, data_types


def _check_dims(data, time = None, n_turns = None):
    """
    Function to check the timebase of data

    Parameters
    ----------
    data : iterable or float
        The data to be checked.
    time : iterable of floats or None, optional
        The time to be used for the array. The default is None.
    n_turns : int or None, optional
        The number of turns to be used for the array. The default is None.

    Raises
    ------
    excpt.InputError
        If the data appears to be turn based and the length does not match
        n_turns an InputError is raised.
        If the time parameter is passed and the data is a 2D array an
        InputError is raised
        If the time parameters is passed and the data is a 1D array of
        different length an InputError is raised
        If the functions reaches the end, the data is not understandable and
        an InputError is raised

    Returns
    -------
    float or iterable of floats
        The data to be used for the new array.
    str
        The timebase of the data.

    """
    #Check and handle single valued data
    #if not single valued coerce to numpy array and continue
    try:
        iter(data)
    except TypeError:
        if n_turns is None and time is None:
            return data, 'single'
        elif n_turns is not None:
            return [data]*n_turns, 'by_turn'
        else:
            return [time, [data]*len(time)], 'by_time'
    else:
        data = np.array(data)

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
    Interpolate time dependant data so all have the same number of points.
    Currently only linear interpolation is available.

    Parameters
    ----------
    data_points : float or iterable of floats
        The data to be interpolated.
    data_types : iterable of str
        The str identifying the timebase of the data.
    interpolation : str, optional
        The interpolation style to be used. The default is 'linear'.

    Raises
    ------
    RuntimeError
        If an interpolation other than linear is requested a RuntimeError is
        raised.
    DataDefinitionError
        If data is passed with timebase other than 'by_time' a 
        DataDefinitionError is raised.

    Returns
    -------
    data_points : List of arrays
        The newly interpolated data.

    """
    if interpolation != 'linear':
        raise RuntimeError("Only linear interpolation defined")

    if all(t == 'single' for t in data_types):
        return data_points

    if data_types[0] != 'by_time':
        excpt.DataDefinitionError("Interpolation only possible if functions "
                                       + "are defined by time")

    input_times = []
    for d in data_points:
        try:
            input_times += d[0].tolist()
        except AttributeError:
            input_times += d[0]

    interp_times = sorted(set(input_times))

    for i in range(len(data_points)):
         interp_data = np.interp(interp_times, data_points[i][0], \
                                 data_points[i][1])
         data_points[i] = np.array([interp_times, interp_data])

    return data_points


def _from_function(inputArray, targetClass, **kwargs):
    """
    WIP
    Function to construct a datatype array from an existing array.

    Parameters
    ----------
    inputArray : Float or iterable
        The data to be checked.
    targetClass : class
        The class to be compared with.
    **kwargs : keyword arguments
        .

    Returns
    -------
    inputArray : The newArray
        DESCRIPTION.

    """
    if isinstance(inputArray, targetClass):
        return inputArray
    
    else:
        return None


def _expand_function(*args):
    """
    Function to expand input functions of the _function or machine_program
    type to remove the extra dimension.

    Parameters
    ----------
    *args : Tuple of functions
        The functions to be checked for type and expanded if necessary.

    Returns
    -------
    tuple
        Tuple of data to be used for the new datatype array.

    """
    newArgs = []
    for a in args:
        if isinstance(a, _function) or isinstance(a, bf.machine_program):
            for sub in a:
                newArgs.append(sub)
        else:
            newArgs.append(a)
    return tuple(newArgs)
