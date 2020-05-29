# General imports
import numpy as np

#Common imports
from ..devtools import exceptions
from ..devtools import assertions as assrt
from . import datatypes as dTypes

class machine_program(np.ndarray):
    
    def __new__(cls, *args, time = None, n_turns = None, 
                interpolation = 'linear'):

        if interpolation != 'linear':
            raise RuntimeError("Only linear interpolation implemented")

        dTypes._check_time_turns(time, n_turns)
        data_points, data_types = dTypes._get_dats_types(*args, time = time, \
                                                  n_turns = n_turns)

        dTypes._check_data_types(data_types, allow_single = True)

        data_types, data_points = dTypes._expand_singletons(data_types, 
                                                                   data_points)

        if not 'by_turn' in data_types:
            data_points = dTypes.interpolate_input(data_points, data_types, 
                                                   interpolation)

        data_type = {'timebase': data_types[0]}

        try:
            obj = np.asarray(data_points).view(cls)
        except ValueError:
            raise exceptions.InputError("Function components could not be " \
                                        + "correctly coerced into ndarray, " \
                                        + "check input dimensionality")

        obj.timebase = data_types[0]

        return obj


    def __array_finalize__(self, obj):

        if obj is None:
            return
        self.timebase = getattr(obj, 'timebase', None)


    def value_at_time(self, time):
        return self.reshape(use_time = time)


    def value_at_turn(self, turn):
        return self.reshape(use_turn = turn)


    def reshape(self, use_time = None, use_turn = None):
        
        assrt.single_not_none(use_time, use_turn, 
                              msg = 'either use_time or use_turns should be'
                              + ' defined, not both', 
                              exception = exceptions.InputError)
        
        if use_time is not None:
            use_points = use_time
        else:
            use_points = np.array(use_turn).astype(int)
        
        nPoints = len(use_points)
        nSects = self.shape[0]
        finalArray = np.zeros([nSects, nPoints])
        for s in range(nSects):
            if self.timebase == 'by_time':
                finalArray[s] = np.interp(use_points, self[s,0], self[s,1])
            elif self.timebase == 'by_turn':
                finalArray[s] = self[s, use_points]
            elif self.timebase == 'single':
                finalArray[s] = self[s]
        
        return finalArray.view(self.__class__)