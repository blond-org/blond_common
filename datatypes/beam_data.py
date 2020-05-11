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
                   _check_turn_numbers, _interpolate_input


class _beam_data(_function):
    """
    Base class defining the data relevant to beam measurements

    Parameters
    ----------
    *args : float, 1D iterable of floats, 2D iterable of floats
        The data defining the programs
    units : str
        The units of the data
    time : iterable of floats
        The time array to be used for the data, to be used if the time 
        dependent data is not given as 2D array
    n_turns : int
        The number of turns to be used, if a single value is given it will be
        extended to n_turns length
    interpolation : str
        The type of interpolation to be used
    **kwargs : keyword arguments
        Additional kwargs used to define the data_type dict.
    
    Attributes
    ----------
    data_type : dict
        Dictionary containing relevant information to define the datatype.
    timebase : str
        Either 'single', 'by_turn', or 'by_time' depending on the definition
        of the datatype.  As a string it is used as a key for the data_type 
        dict.
    units : str
        Identifies the units of the stored data.  Used as a string key in the
        data_type dict.
    interpolation : str
        Identifier of the type of interpolation to be used when reshaping
        the array, currently only 'linear' has been implemented.
    """
    
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
            data_points = _interpolate_input(data_points, data_types, 
                                             interpolation)

        if len(data_types) == 1:
            bunching = 'single_bunch'
        else:
            bunching = 'multi_bunch'
        
        data_type = {'timebase': data_types[0], 'bunching': bunching,
                     'units': units, **kwargs}
        
        return super().__new__(cls, data_points, data_type, interpolation)


    @property
    def bunching(self):
        """
        Get and set bunching attribute.  The data_type dict will be updated
        when the attribute is set.
        """
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
        """
        Get and set units attribute.  The data_type dict will be updated
        when the attribute is set.
        """
        try:
            return self._units
        except AttributeError:
            return None
    
    @units.setter
    def units(self, value):
        self._check_data_type('units', value)
        self._units = value


class acceptance(_beam_data):
    """
    Class defining the acceptance of the RF bucket.

    Parameters
    ----------
    *args : float, 1D iterable of floats, 2D iterable of floats
        The data defining the programs
    units : str, optional
        The units of the data.  The default is 'eVs'
    time : iterable of floats, optional
        The time array to be used for the data, to be used if the time 
        dependent data is not given as 2D array.  The default is None
    n_turns : int, option
        The number of turns to be used, if a single value is given it will be
        extended to n_turns length.  The default is None.
    interpolation : str, optional
        The type of interpolation to be used.  The default is 'linear'.
    **kwargs : keyword arguments
        Additional kwargs used to define the data_type dict.
    
    Attributes
    ----------
    data_type : dict
        Dictionary containing relevant information to define the datatype.
    timebase : str
        Either 'single', 'by_turn', or 'by_time' depending on the definition
        of the datatype.  As a string it is used as a key for the data_type 
        dict.
    units : str
        Identifies the units of the stored data.  Used as a string key in the
        data_type dict.
    interpolation : str
        Identifier of the type of interpolation to be used when reshaping
        the array, currently only 'linear' has been implemented.
    """
    
    def __new__(cls, *args, units = 'eVs', time = None, n_turns = None, 
                interpolation = 'linear'):
        
        return super().__new__(cls, *args, units = units, time = time, 
                               n_turns = n_turns, 
                               interpolation = interpolation)


class emittance(_beam_data):
    """
    Class defining the longitudinal emittance of the bunch.

    Parameters
    ----------
    *args : float, 1D iterable of floats, 2D iterable of floats
        The data defining the programs
    emittance_type : str, optional
        The emittance type of the measurement (e.g. 90%, matched area, RMS).
        The default is 'matched_area'.
    units : str, optional
        The units of the data.  The default is 'eVs'
    time : iterable of floats, optional
        The time array to be used for the data, to be used if the time 
        dependent data is not given as 2D array.  The default is None
    n_turns : int, option
        The number of turns to be used, if a single value is given it will be
        extended to n_turns length.  The default is None.
    interpolation : str, optional
        The type of interpolation to be used.  The default is 'linear'.
    
    Attributes
    ----------
    data_type : dict
        Dictionary containing relevant information to define the datatype.
    timebase : str
        Either 'single', 'by_turn', or 'by_time' depending on the definition
        of the datatype.  As a string it is used as a key for the data_type 
        dict.
    emittance_type : str
        Identifies the type of emittance defined by the data (e.g. 90%, 
                                                              matched area
                                                              RMS).  Used as
        a string key in the data_type dict.
    units : str
        Identifies the units of the stored data.  Used as a string key in the
        data_type dict.
    interpolation : str
        Identifier of the type of interpolation to be used when reshaping
        the array, currently only 'linear' has been implemented.
    """
    
    def __new__(cls, *args, emittance_type = 'matched_area', units = 'eVs', 
                time = None, n_turns = None, interpolation = 'linear'):
        
        return super().__new__(cls, *args, 
                               emittance_type = emittance_type,
                               units = units, time = time, 
                               n_turns = n_turns,
                               interpolation = interpolation)


    @property
    def emittance_type(self):
        """
        Get and set emittance_type.  When set the data_type dict is also 
        updated
        """
        try:
            return self._emittance_type
        except AttributeError:
            return None
    
    @emittance_type.setter
    def emittance_type(self, value):
        self._check_data_type('emittance_type', value)
        self._emittance_type = value


class length(_beam_data):
    """
    Class defining the bunch length.

    Parameters
    ----------
    *args : float, 1D iterable of floats, 2D iterable of floats
        The data defining the programs
    length_type : str, optional
        The emittance type of the measurement (e.g. 4 sigma, full length, RMS).
        The default is 'matched_area'.
    units : str, optional
        The units of the data.  The default is 's'
    time : iterable of floats, optional
        The time array to be used for the data, to be used if the time 
        dependent data is not given as 2D array.  The default is None
    n_turns : int, option
        The number of turns to be used, if a single value is given it will be
        extended to n_turns length.  The default is None.
    interpolation : str, optional
        The type of interpolation to be used.  The default is 'linear'.
    
    Attributes
    ----------
    data_type : dict
        Dictionary containing relevant information to define the datatype.
    timebase : str
        Either 'single', 'by_turn', or 'by_time' depending on the definition
        of the datatype.  As a string it is used as a key for the data_type 
        dict.
    length_type : str
        Identifies the type of emittance defined by the data (e.g. 4 sigma, 
                                                              full length, 
                                                              RMS).  Used as
        a string key in the data_type dict.
    units : str
        Identifies the units of the stored data.  Used as a string key in the
        data_type dict.
    interpolation : str
        Identifier of the type of interpolation to be used when reshaping
        the array, currently only 'linear' has been implemented.
    """
    
    def __new__(cls, *args, length_type = 'full_length', units = 's',
                time = None, n_turns = None, interpolation = 'linear'):
        
        return super().__new__(cls, *args, length_type = length_type, 
                               units = units, time = time, 
                               n_turns = n_turns,
                               interpolation = interpolation)


    @property
    def length_type(self):
        """
        Get and set length_type attribute.  On set the data_type dict will
        also be updated
        """
        try:
            return self._length_type
        except AttributeError:
            return None
    
    @length_type.setter
    def length_type(self, value):
        self._check_data_type('length_type', value)
        self._length_type = value
        

class height(_beam_data):
    """
    Class defining the bunch height in energy.

    Parameters
    ----------
    *args : float, 1D iterable of floats, 2D iterable of floats
        The data defining the programs
    height_type : str, optional
        The emittance type of the measurement (e.g. half height, full height).
        The default is 'half height'.
    units : str, optional
        The units of the data.  The default is 'eV'
    time : iterable of floats, optional
        The time array to be used for the data, to be used if the time 
        dependent data is not given as 2D array.  The default is None
    n_turns : int, option
        The number of turns to be used, if a single value is given it will be
        extended to n_turns length.  The default is None.
    interpolation : str, optional
        The type of interpolation to be used.  The default is 'linear'.
    
    Attributes
    ----------
    data_type : dict
        Dictionary containing relevant information to define the datatype.
    timebase : str
        Either 'single', 'by_turn', or 'by_time' depending on the definition
        of the datatype.  As a string it is used as a key for the data_type 
        dict.
    height_type : str
        Identifies the type of emittance defined by the data (e.g. half height,
                                                              full height).  
        Used as a string key in the data_type dict.
    units : str
        Identifies the units of the stored data.  Used as a string key in the
        data_type dict.
    interpolation : str
        Identifier of the type of interpolation to be used when reshaping
        the array, currently only 'linear' has been implemented.
    """
    
    def __new__(cls, *args, height_type = 'half_height', units = 'eV',
                time = None, n_turns = None, interpolation = 'linear'):
        
        return super().__new__(cls, *args, height_type = height_type, 
                               units = units, time = time, n_turns = n_turns,
                               interpolation = interpolation)


    @property
    def height_type(self):
        """
        Get and set emittance_type.  When set the data_type dict is also 
        updated
        """
        try:
            return self._height_type
        except AttributeError:
            return None
    
    @height_type.setter
    def height_type(self, value):
        self._check_data_type('height_type', value)
        self._height_type = value


class synchronous_phase(_beam_data):
    """
    Class defining the synchronous phase.

    Parameters
    ----------
    *args : float, 1D iterable of floats, 2D iterable of floats
        The data defining the programs
    units : str, optional
        The units of the data.  The default is 's'
    time : iterable of floats, optional
        The time array to be used for the data, to be used if the time 
        dependent data is not given as 2D array.  The default is None
    n_turns : int, option
        The number of turns to be used, if a single value is given it will be
        extended to n_turns length.  The default is None.
    interpolation : str, optional
        The type of interpolation to be used.  The default is 'linear'.
    
    Attributes
    ----------
    data_type : dict
        Dictionary containing relevant information to define the datatype.
    timebase : str
        Either 'single', 'by_turn', or 'by_time' depending on the definition
        of the datatype.  As a string it is used as a key for the data_type 
        dict.
    units : str
        Identifies the units of the stored data.  Used as a string key in the
        data_type dict.
    interpolation : str
        Identifier of the type of interpolation to be used when reshaping
        the array, currently only 'linear' has been implemented.
    """
    
    def __new__(cls, *args, units = 's', time = None, n_turns = None, 
                interpolation = 'linear'):
        
        return super().__new__(cls, *args, units = units, time = time, 
                               n_turns = n_turns, 
                               interpolation = interpolation)
