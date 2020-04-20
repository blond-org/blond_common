# General imports
import numpy as np

# BLonD_Common imports
from ...datatypes import datatypes as dTypes
from ...datatypes import blond_function as bf
from ...devtools import exceptions as excpt
from ...devtools import assertions as assrt


class ring_section:
    
    def __init__(self, section_length, alpha, momentum = None,
                 kin_energy = None, energy = None, bending_field = None,
                 bending_radius = None):

        self.section_length = section_length

        syncDataTypes = ('momentum', 'kin_energy', 'energy', 
                         'B_field')
        syncDataInput = (momentum, kin_energy, energy, bending_field)
        assrt.single_not_none(*syncDataInput, msg = 'Exactly one of '
                              + str(syncDataTypes) + ' must be declared',
                              exception = excpt.InputError)

        if bending_field is not None and bending_radius is None:
            raise excpt.InputError("If bending_field is used, bending_radius "
                                   + "must be defined.")

        for t, i in zip(syncDataTypes, syncDataInput):
            if i is not None:
                func_type = t
                synchronous_data = i
                break

        # Reshaping the input synchronous data to the adequate format and
        # get back the momentum program from RingOptions
        if not isinstance(synchronous_data, dTypes._ring_program):
            synchronous_data \
                = dTypes._ring_program.conversions[func_type](synchronous_data)

        self.synchronous_data = synchronous_data

        if bending_radius is not None:
            self.bending_radius = float(bending_radius)
        else:
            self.bending_radius = bending_radius


        if not hasattr(alpha, '__iter__'):
            alpha = (alpha, )
        
        if isinstance(alpha, dict):
            try:
                if not all([k%1 == 0 for k in alpha.keys()]):
                    raise TypeError
            except TypeError:
                raise excpt.InputError("If alpha is dict all keys must be "
                                       + "numeric and integer")

            maxAlpha = np.max(tuple(alpha.keys())).astype(int)
            alpha = [alpha.pop(i, 0) for i in range(maxAlpha+1)]
        
        if isinstance(alpha, dTypes._function) or \
            isinstance(alpha, bf.machine_program):
            alpha = (alpha,)

        for i, a in enumerate(alpha):
            if not isinstance(a, dTypes.momentum_compaction):
                a = dTypes.momentum_compaction(a, order = i)
            setattr(self, 'alpha_'+str(i), a)
        self.alpha_order = i
        
        for i in range(3 - self.alpha_order):
            if not hasattr(self, f'alpha_{i}'):
                setattr(self, 'alpha_'+str(i), dTypes.momentum_compaction(0,
                                                                  order = i))
