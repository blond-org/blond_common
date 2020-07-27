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


class _RF_function(_function):
    """
    Base class for defining functions used for the RFStation

    Parameters
    ----------
    *args : float, 1D iterable of floats, 2D iterable of floats
        The data defining the programs.
    harmonics : iterable of ints
        The harmonics covered by the data.
    time : iterable of floats
        The time array to be used for the data, to be used if the time 
        dependent data is not given as 2D array.
    n_turns : int
        The number of turns to be used, if a single value is given it will be
        extended to n_turns length.
    interpolation : str
        The type of interpolation to be used.
    **kwargs : keyword arguments
        Additional kwargs used to define the data_type dict.
    
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
    harmonics : iterable of ints
        The harmonics covered by the function.
    """
    def __new__(cls, *args, harmonics, time = None, n_turns = None, \
                interpolation = 'linear', allow_single = True, dtype = None,
                **kwargs):

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
            data_points = _interpolate_input(data_points, data_types, 
                                            interpolation)

        data_type = {'timebase': data_types[0], 'harmonics': harmonics, 
                     **kwargs}

        return super().__new__(cls, data_points, data_type, 
                               interpolation, dtype)


    @property
    def harmonics(self):
        """
        Get or set the harmonics attribute.  Setting the harmonics will also
        update the data_type dict.
        """
        try:
            return self._harmonics
        except AttributeError:
            return None
    
    @harmonics.setter
    def harmonics(self, value):
        self._check_data_type('harmonics', value)
        self._harmonics = value


    #TODO: Safe treatment of use_turns > n_turns
    def reshape(self, harmonics = None, use_time = None, use_turns = None,
                store_time = False):
        """
        Reshape the datatype array to the given number of sections and either
        the given use_time or given use_turns.

        Parameters
        ----------
        harmonics : iterable of ints, optional
            The harmonics to be returned in the new array.  If none all 
            defined harmonics will be returned.  All specified harmonics will
            be returned, if not defined in the function they will be 0.
            The default is None.
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
        if harmonics is None:
            harmonics = self.harmonics

        if use_turns is not None:
            use_turns = [int(turn) for turn in use_turns]

        newArray = self._prep_reshape(len(harmonics), 
                                      use_time = use_time,
                                      use_turns = use_turns,
                                      store_time = store_time)
        
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

            else:
                raise RuntimeError("Only single, by_turn or by_time functions"
                                   +f" can be reshaped, not {self.timebase}.")

        newArray = newArray.view(self.__class__)

        if store_time:
            newArray[:, 0, :] = use_time
            newArray.timebase = 'by_time'
        else:
            newArray.timebase = 'interpolated'

        newArray.harmonics = harmonics

        return newArray


class voltage_program(_RF_function):
    """
    Class for defining voltage functions.

    Parameters
    ----------
    *args : float, 1D iterable of floats, 2D iterable of floats
        The data defining the programs.
    harmonics : iterable of ints
        The harmonics covered by the data.
    time : iterable of floats
        The time array to be used for the data, to be used if the time 
        dependent data is not given as 2D array.
    n_turns : int
        The number of turns to be used, if a single value is given it will be
        extended to n_turns length.
    interpolation : str
        The type of interpolation to be used.
    **kwargs : keyword arguments
        Additional kwargs used to define the data_type dict.
    
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
    harmonics : iterable of ints
        The harmonics covered by the function.
    """
    pass


class phase_program(_RF_function):
    """
    Class for defining phase functions.

    Parameters
    ----------
    *args : float, 1D iterable of floats, 2D iterable of floats
        The data defining the programs.
    harmonics : iterable of ints
        The harmonics covered by the data.
    time : iterable of floats
        The time array to be used for the data, to be used if the time 
        dependent data is not given as 2D array.
    n_turns : int
        The number of turns to be used, if a single value is given it will be
        extended to n_turns length.
    interpolation : str
        The type of interpolation to be used.
    **kwargs : keyword arguments
        Additional kwargs used to define the data_type dict.
    
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
    harmonics : iterable of ints
        The harmonics covered by the function.
    """
    pass


class _freq_phase_off(_RF_function):
    """
    Class for defining offsets to the design RF phase and frequency.

    Parameters
    ----------
    *args : float, 1D iterable of floats, 2D iterable of floats
        The data defining the programs.
    harmonics : iterable of ints
        The harmonics covered by the data.
    time : iterable of floats
        The time array to be used for the data, to be used if the time 
        dependent data is not given as 2D array.
    n_turns : int
        The number of turns to be used, if a single value is given it will be
        extended to n_turns length.
    interpolation : str
        The type of interpolation to be used.
    **kwargs : keyword arguments
        Additional kwargs used to define the data_type dict.
    
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
    harmonics : iterable of ints
        The harmonics covered by the function.
    """
    
    def calc_delta_omega(self, design_omega_rev):
        r"""
        Calculare the change in $\Omega_{RF}$ required to give a given phase
        offset.

        Parameters
        ----------
        design_omega_rev : 1D or 2D array.
            The design $\Omega_{0}$ used to compute the $\Delta\Omega_{RF}$.  
            Taken as turn dependent if 1D or time dependent if 2D.

        Raises
        ------
        RuntimeError
            If this function is called with an omega_offset function a 
            RuntimeError is raised.
            If timebase == 'by_turn' ('by_time') and design_omega_rev is a 
            time based (turn based) function a RuntimeError is raised.
        excpt.InputError
            If the shape of the data and the design_omega_rev are mismatched
            and InputError is raised.

        Returns
        -------
        delta_omega : omega_offset
            The $\Delta\Omega_{RF}$ at each defined harmonic.
        """
        
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
                delta_omega[i] = h*(design_omega_rev/(2*np.pi)) \
                                    * np.gradient(delta_phase[i])
            else:
                delta_omega[i] = h*(design_omega_rev[1]/(2*np.pi)) \
                                    * np.gradient(delta_phase[i])\
                                        /np.gradient(design_omega_rev[0])
        
        return delta_omega
    
    
    def calc_delta_phase(self, design_omega_rev, wrap=False):
        r"""
        Calculare the change in $\Omega_{RF}$ required to give a given phase
        offset.

        Parameters
        ----------
        design_omega_rev : 1D or 2D array.
            The design $\Omega_{0}$ used to compute the $\Delta\varphi_{RF}$.  
            Taken as turn dependent if 1D or time dependent if 2D.
        wrap : bool
            WIP:  To be used to flag the modulus of the phase offset

        Raises
        ------
        RuntimeError
            If this function is called with an omega_offset function a 
            RuntimeError is raised.
            If timebase == 'by_turn' ('by_time') and design_omega_rev is a 
            time based (turn based) function a RuntimeError is raised.
        excpt.InputError
            If the shape of the data and the design_omega_rev are mismatched
            and InputError is raised.

        Returns
        -------
        delta_phase : phase_offset
            The $\Delta\varphi_{RF}$ at each defined harmonic.
        """

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
    """
    Class for defining phase offset functions used for the RFStation

    Parameters
    ----------
    *args : float, 1D iterable of floats, 2D iterable of floats
        The data defining the programs.
    harmonics : iterable of ints
        The harmonics covered by the data.
    time : iterable of floats
        The time array to be used for the data, to be used if the time 
        dependent data is not given as 2D array.
    n_turns : int
        The number of turns to be used, if a single value is given it will be
        extended to n_turns length.
    interpolation : str
        The type of interpolation to be used.
    **kwargs : keyword arguments
        Additional kwargs used to define the data_type dict.
    
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
    harmonics : iterable of ints
        The harmonics covered by the function.
    """
    pass

class omega_offset(_freq_phase_off):
    """
    Class for defining phase offset functions used for the RFStation

    Parameters
    ----------
    *args : float, 1D iterable of floats, 2D iterable of floats
        The data defining the programs.
    harmonics : iterable of ints
        The harmonics covered by the data.
    time : iterable of floats
        The time array to be used for the data, to be used if the time 
        dependent data is not given as 2D array.
    n_turns : int
        The number of turns to be used, if a single value is given it will be
        extended to n_turns length.
    interpolation : str
        The type of interpolation to be used.
    **kwargs : keyword arguments
        Additional kwargs used to define the data_type dict.
    
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
    harmonics : iterable of ints
        The harmonics covered by the function.
    """
    pass