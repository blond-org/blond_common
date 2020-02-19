# coding: utf8
# Copyright 2019 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Module to interpolate time arrays
:Authors: **Simon Albright**
"""

#General imports
import numpy as np
import warnings
import numbers

#BLonD_Common imports
from ..devtools import exceptions as excpt


def time_from_sampling(resolution):
    
    if resolution == 't_rev':

        def sample_func(time):
            return time
        
        start = 0
        end = np.inf
    
    elif isinstance(resolution, numbers.Number):

        def sample_func(time):
            return time + resolution
        
        start = 0
        end = np.inf
    
    elif isinstance(resolution, tuple):
        
        def sample_func(time):
            if time >= resolution[1][0] and time < resolution[1][1]:
                return time + resolution[0]

            else:
                return np.inf
        
        start, end = resolution[1]
                
    elif isinstance(resolution, list):
        
        def sample_func(time):
            for r in resolution:
                if time >= r[1][0] and time < r[1][1]:
                    return time + r[0]
            else:
                next_time = np.inf
                for r in resolution:
                    if time <= r[1][0] and r[1][0] < next_time:
                        next_time = r[1][0]

                return next_time
                
        start = np.inf
        end = 0
        for r in resolution:
            if r[1][0] < start:
                start = r[1][0]
            if r[1][1] > end:
                end = r[1][1]

    return sample_func, start, end


# Calculate turn numbers from a passed time_range at times given by 'resolution'
# If resolution is a float, turn numbers evenly spaced by 'resolution' s will
# be returned.
# If resolution is a tuple the first element is taken as the time spacing and
# the second element should be a length 2 iterable of start/stop time 
# for interpolation
# If resolution is a list, each element of the list is treated the same as if
# a tuple had been passed
def points_by_time(time_range, resolution = 1E-3):
    
    if isinstance(resolution, float):
        cycle_points = time_points(time_range, resolution)

    elif isinstance(resolution, tuple):
        cycle_points = time_points(time_range, resolution[0],
                                   resolution[1][0], resolution[1][1])

    elif isinstance(resolution, list):
        cycle_points = []
        for i in range(len(resolution)):
            if len(cycle_points) > 0:
                if (resolution[i][1][0] <
                    time_range[cycle_points[-1]] + resolution[i][0]):

                    resolution[i][1][0] = (time_range[cycle_points[-1]]
                                           + resolution[i][0])

            cycle_points += time_points(time_range, resolution[i][0],
                              resolution[i][1][0], resolution[i][1][1]).tolist()

    else:
        raise excpt.InputError("resolution must be float, tuple or list")

    return np.asarray(cycle_points)


#From start:stop identify indices of time_range with 'resolution' separation.
def time_points(time_range, resolution = 1E-3, start = None, stop = None):
    
    if start is None:
        start = time_range[0]
    if stop is None:
        stop = time_range[-1]

    if start < time_range[0]:
        warnings.warn("Start time before cycle starts,"+
                " defaulting to start")
        start = time_range[0]
    if stop > time_range[-1]:
        warnings.warn("Stop time after cycle ends,"+
                " defaulting to cycle end")
        stop = time_range[-1]

    point = 0

    while time_range[point] < start:
        point += 1
        
    pointList = [point]

    while time_range[point] < stop:
        if (time_range[point] >=
            time_range[pointList[-1]]
            + resolution):
               pointList.append(point)
        point += 1
    pointList.append(point-1)

    return np.asarray(pointList)