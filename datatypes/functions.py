# General imports
import numpy as np
import sys
import os
import warnings
import scipy.constants as cont
import matplotlib.pyplot as plt

# BLonD_Common imports
from ..devtools import exceptions as excpt
from ..devtools import assertions as assrt
from ..utilities import rel_transforms as rt
from . import blond_function as bf
from ._core import _function

def vstack(*args, interpolation = 'linear'):
    """
    Vertically stack the datatype arrays.  The input can include the start
    and stop time of each datatype.

    Parameters
    ----------
    *args : tuples or datatypes
        The arrays to be stacked.
    interpolation : str, optional
        The type of interpolation to be used for the new datatype array.
        The default is 'linear'.

    Raises
    ------
    excpt.InputError
        If the inputs are not all iterable an InputError is raised.
        If the inputs is a tuple and does not include a datatype array an
        InputError is raised.
        If a tuple is passed and it has length of more than 3 an InputError is
        raised.
        If the functions are not all datatype arrays an InputError is raised.
        If a single valued datatype is passed an InputError is raised.
        If the datatype arrays have different timebases an InputError is 
        raised.
        If the datatypes do not have the same data_type dict an InputError 
        is raised.
        
    Returns
    -------
    newArray : datatype
        The stacked array.

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