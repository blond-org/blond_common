# coding: utf-8
# Copyright 2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Module to compute the beam profile through slices**

:Authors: **Danilo Quartullo**, **Alexandre Lasheen**, 
          **Juan F. Esteban Mueller**, **Simon Albright**
'''

# General imports
from __future__ import division, print_function
from builtins import object
import numpy as np
import numpy.fft as fft

# BLonD_common imports
from ...devtools import assertions as assrt
from ...devtools import exceptions as excpt

class CutOptions:
    r"""
    This class groups all the parameters necessary to slice the phase space
    distribution according to the time axis, apart from the array collecting
    the profile which is defined in the constructor of the class Profile below.

    Parameters
    ----------
    cut_left : float
        Left edge of the slicing (optional). A default value will be set if
        no value is given.
    cut_right : float
        Right edge of the slicing (optional). A default value will be set
        if no value is given.
    n_slices : int
        Optional input parameters, corresponding to the number of
        :math:`\sigma_{RMS}` of the Beam to slice (this will overwrite
        any input of cut_left and cut_right).
    n_sigma : float
        defines the left and right extremes of the profile in case those are
        not given explicitly
    cuts_unit : str
        the unit of cut_left and cut_right, it can be seconds 's' or radians
        'rad'
    RFSectionParameters : object
        RFSectionParameters[0][0] is necessary for the conversion from radians
        to seconds if cuts_unit = 'rad'. RFSectionParameters[0][0] is the value
        of omega_rf of the main harmonic at turn number 0

    Attributes
    ----------
    cut_left : float
    cut_right : float
    n_slices : int
    n_sigma : float
    cuts_unit : str
    RFSectionParameters : object
    edges : float array
        contains the edges of the slices
    bin_centers : float array
        contains the centres of the slices

    Examples
    --------
    >>> from input_parameters.ring import Ring
    >>> from input_parameters.rf_parameters import RFStation
    >>> self.ring = Ring(n_turns = 1, ring_length = 100,
    >>> alpha = 0.00001, momentum = 1e9)
    >>> self.rf_params = RFStation(Ring=self.ring, n_rf=1, harmonic=[4620],
    >>>                  voltage=[7e6], phi_rf_d=[0.])
    >>> CutOptions = profileModule.CutOptions(cut_left=0, cut_right=2*np.pi,
    >>> n_slices = 100, cuts_unit='rad', RFSectionParameters=self.rf_params)

    """

    def __init__(self, cut_left=None, cut_right=None, n_slices=100,
                 n_sigma=None, cuts_unit='s', RFSectionParameters=None):
        """
        Constructor
        """

        if cut_left is not None:
            self.cut_left = float(cut_left)
        else:
            self.cut_left = cut_left

        if cut_right is not None:
            self.cut_right = float(cut_right)
        else:
            self.cut_right = cut_right

        self.n_slices = int(n_slices)

        if n_sigma is not None:
            self.n_sigma = float(n_sigma)
        else:
            self.n_sigma = n_sigma

        self.cuts_unit = str(cuts_unit)

        self.RFParams = RFSectionParameters

        if self.cuts_unit == 'rad' and self.RFParams is None:
            #CutError
            raise RuntimeError('You should pass an RFParams object to ' +
                               'convert from radians to seconds')
        if self.cuts_unit != 'rad' and self.cuts_unit != 's':
            #CutError
            raise RuntimeError('cuts_unit should be "s" or "rad"')

        self.edges = np.zeros(n_slices + 1, dtype=float)
        self.bin_centers = np.zeros(n_slices, dtype=float)

    def set_cuts(self, Beam=None):
        """
        Method to set self.cut_left, self.cut_right, self.edges and
        self.bin_centers attributes.
        The frame is defined by :math:`n\sigma_{RMS}` or manually by the user.
        If not, a default frame consisting of taking the whole bunch +5% of the
        maximum distance between two particles in the bunch will be taken
        in each side of the frame.
        """

        if self.cut_left is None and self.cut_right is None:

            if self.n_sigma is None:
                dt_min = Beam.dt.min()
                dt_max = Beam.dt.max()
                self.cut_left = dt_min - 0.05 * (dt_max - dt_min)
                self.cut_right = dt_max + 0.05 * (dt_max - dt_min)
            else:
                mean_coords = np.mean(Beam.dt)
                sigma_coords = np.std(Beam.dt)
                self.cut_left = mean_coords - self.n_sigma*sigma_coords/2
                self.cut_right = mean_coords + self.n_sigma*sigma_coords/2

        else:

            self.cut_left = float(self._convert_coordinates(self.cut_left,
                                                     self.cuts_unit))
            self.cut_right = float(self._convert_coordinates(self.cut_right,
                                                      self.cuts_unit))

        self.edges = np.linspace(self.cut_left, self.cut_right,
                                 self.n_slices + 1)
        self.bin_centers = (self.edges[:-1] + self.edges[1:])/2
        self.bin_size = (self.cut_right - self.cut_left) / self.n_slices

    def track_cuts(self, Beam):
        """
        Track the slice frame (limits and slice position) as the mean of the
        bunch moves.
        Requires Beam statistics!
        Method to be refined!
        """

        delta = Beam.mean_dt - 0.5*(self.cut_left + self.cut_right)

        self.cut_left += delta
        self.cut_right += delta
        self.edges += delta
        self.bin_centers += delta

    def _convert_coordinates(self, value, input_unit_type):
        """
        Method to convert a value from 'rad' to 's'.
        """

        if input_unit_type == 's':
            return value

        elif input_unit_type == 'rad':
            return value /\
                self.RFParams.omega_rf[0, self.RFParams.counter[0]]
        
        else:
            raise excpt.InputError("input_unit_type not recognised")
        

    def get_slices_parameters(self):
        """
        Reuturn all the computed parameters.
        """
        return self.n_slices, self.cut_left, self.cut_right, self.n_sigma, \
            self.edges, self.bin_centers, self.bin_size


class FitOptions:
    """
    This class defines the method to be used turn after turn to obtain the
    position and length of the bunch profile.

    Parameters
    ----------

    fit_method : string
        Current options are 'gaussian',
        'fwhm' (full-width-half-maximum converted to 4 sigma gaussian bunch)
        and 'rms'. The methods 'gaussian' and 'rms' give both 4 sigma.
    fitExtraOptions : unknown
        For the moment no options can be passed into fitExtraOptions

    Attributes
    ----------

    fit_method : string
    fitExtraOptions : unknown
    """

    def __init__(self, fit_option=None, fitExtraOptions=None):

        """
        Constructor
        """

        self.fit_option = str(fit_option)
        self.fitExtraOptions = fitExtraOptions


class FilterOptions:

    """
    This class defines the filter to be used turn after turn to smooth
    the bunch profile.

    Parameters
    ----------

    filterMethod : string
        The only option available is 'chebishev'
    filterExtraOptions : dictionary
        Parameters for the Chebishev filter (see the method
        beam_profile_filter_chebyshev in filters_and_fitting.py in the toolbox
        package)

    Attributes
    ----------

    filterMethod : string
    filterExtraOptions : dictionary

    """

    def __init__(self, filterMethod=None, filterExtraOptions=None):

        """
        Constructor
        """

        self.filterMethod = str(filterMethod)
        self.filterExtraOptions = filterExtraOptions


class OtherSlicesOptions:

    """
    This class groups all the remaining options for the Profile class.

    Parameters
    ----------

    smooth : boolean
        If set True, this method slices the bunch not in the
        standard way (fixed one slice all the macroparticles contribute
        with +1 or 0 depending if they are inside or not). The method assigns
        to each macroparticle a real value between 0 and +1 depending on its
        time coordinate. This method can be considered a filter able to smooth
        the profile.
    direct_slicing : boolean
        If set True, the profile is calculated when the Profile class below
        is created. If False the user has to manually track the Profile object
        in the main file after its creation

    Attributes
    ----------

    smooth : boolean
    direct_slicing : boolean

    """

    def __init__(self, smooth=False, direct_slicing=False):

        """
        Constructor
        """

        self.smooth = smooth
        self.direct_slicing = direct_slicing


class Profile:
    
    def __init__(self, time_array, profile_array):
        
        assrt.equal_array_lengths(time_array, profile_array,
                  msg = 'time_array and profile_array should be equal length',
                  exception = excpt.InputDataError)
        
        self.time_array_loaded = time_array
        self.profile_array_loaded = profile_array
        
        self.bin_size = time_array[1] - time_array[0]
        
        #Duplicating initially, will be replaced by any smoothing functions
        self.time_array = self.time_array_loaded.copy()
        self.profile_array = self.profile_array_loaded.copy()
        
    
    def beam_spectrum_freq_generation(self, n_sampling_fft = None):
        """
        Frequency array of the beam spectrum
        """
        if n_sampling_fft is None:
            n_sampling_fft = len(self.profile_array)
        self._beam_spectrum_freq = fft.rfftfreq(n_sampling_fft, self.bin_size)

    def beam_spectrum_generation(self, n_sampling_fft = None):
        """
        Beam spectrum calculation
        """

        self._beam_spectrum = fft.rfft(self.profile_array, n_sampling_fft)
    
    @property
    def beam_spectrum_freq(self):
        
        try:
            return self._beam_spectrum_freq
        except AttributeError:
            self.beam_spectrum_freq_generation()
            return self._beam_spectrum_freq
        
    @property
    def beam_spectrum(self):
        
        try:
            return self._beam_spectrum
        except AttributeError:
            self.beam_spectrum_generation()
            return self._beam_spectrum