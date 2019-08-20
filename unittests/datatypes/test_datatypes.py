# coding: utf8
# Copyright 2019 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Test preprocess.py

'''

# General imports
# ---------------
import sys
import unittest
import numpy as np
import os

#BLonD_Common imports
#----------------
this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
if os.path.abspath(this_directory + '../../../') not in sys.path:
    sys.path.insert(0, os.path.abspath(this_directory + '../../../'))

import blond_common.datatypes.datatypes as dtypes
import blond_common.utilities.exceptions as exceptions

class test_datatypes(unittest.TestCase):
    
    def test__function_exceptions(self):
        
        with self.assertRaises(exceptions.InputError, \
                               msg='Undeclared data_type should raise InputError'):
            dtypes._function(1)
        
        test1 = np.array([[1, 2, 3], [4, 5, 6]])
        test2 = np.array([[1, 2], [4, 5]])
        
        with self.assertRaises(exceptions.InputError, \
                               msg='Misshaped input should raise InputError'):
            dtypes._function([test1, test2], 'test')


    def test__function_data_types(self):
        
        test = np.array([1, 2, 3])
        out1 = dtypes._function(test, 'abc')
        out2 = dtypes._function(test, 123)
        out3 = dtypes._function(test, ('abc', 123))
        
        self.assertEqual(out1.data_type, 'abc', \
                         'data_type not assigned correctly')
        self.assertEqual(out2.data_type, 123, \
                         'data_type not assigned correctly')
        self.assertEqual(out3.data_type, ('abc', 123), \
                         'data_type not assigned correctly')


    def test__function_array_like(self):
        
        test = np.array([1, 2, 3])
        out1 = dtypes._function(test, 1)
        self.assertIsInstance(out1, np.ndarray, '_function is not array')


    def test_momentum_data_type(self):
        
        inArr = np.array([[1, 2, 3], [4, 5, 6]])

        momentum = dtypes.momentum_program(inArr)
        
        self.assertIsInstance(momentum, dtypes.momentum_program, \
                              'momentum should be momentum_program type')

        inDict = {}
        inDict[('momentum', 'by_time', 'single_section')] = (inArr,)
        inDict[('momentum', 'by_time', 'multi_section')] = (inArr, inArr)
        inDict[('momentum', 'by_turn', 'single_section')] = (inArr[0],)
        inDict[('momentum', 'by_turn', 'multi_section')] = (inArr[0], inArr[1])
        inDict[('momentum', 'by_turn', 'single_section')] = (inArr[0],)
        inDict[('momentum', 'by_turn', 'single_section')] = ([1, 2, 3],)
        
        for item in inDict:
            momentum = dtypes.momentum_program(*inDict[item])
            self.assertEqual(momentum.data_type, item, \
                             'momentum identified incorrect with ' + str(item) \
                             + str(inDict[item]))
        
        momentum = dtypes.momentum_program(1, n_turns = 5)

        self.assertEqual(momentum.data_type, ('momentum', 'by_turn', 'single_section'), \
                        'momentum data_type identifier incorrect')
        
        momentum = dtypes.momentum_program(1, 2, 3, n_turns = 5)

        self.assertEqual(momentum.data_type, ('momentum', 'by_turn', 'multi_section'), \
                        'momentum data_type identifier incorrect')
        
        momentum = dtypes.momentum_program(1)

        self.assertEqual(momentum.data_type, ('momentum', 'single', 'single_section'), \
                        'momentum data_type identifier incorrect')

        momentum = dtypes.momentum_program(1, 2, 3)

        self.assertEqual(momentum.data_type, ('momentum', 'single', 'multi_section'), \
                        'momentum data_type identifier incorrect')
        
        momentum = dtypes.momentum_program([1, 2, 3], time=[1,2 ,3])

        self.assertEqual(momentum.data_type, ('momentum', 'by_time', 'single_section'), \
                        'momentum data_type identifier incorrect')
        
    
    def test_RF_section_function_data_type(self):
        
        inArr = np.array([[1, 2, 3], [4, 5, 6]])

        dtypes.RF_section_function(inArr, harmonics = 1)
        
        inDict = {}
        inDict[('RF', 'by_time', (1,))] = (inArr,)
        inDict[('RF', 'by_time', (1, 2))] = (inArr, inArr)
        inDict[('RF', 'by_turn', (1,))] = (inArr[0],)
        inDict[('RF', 'by_turn', (1, 2))] = (inArr[0], inArr[1])
        inDict[('RF', 'by_turn', (1,))] = (inArr[0],)
        inDict[('RF', 'by_turn', (1,))] = ([1, 2, 3],)
        
        for item in inDict:
            RF = dtypes.RF_section_function(*inDict[item], harmonics=item[2])
            self.assertEqual(RF.data_type, item, \
                             'momentum identified incorrect with ' + str(item) \
                             + str(inDict[item]))
        
        RF = dtypes.RF_section_function(1, 2, harmonics = [1, 2])
        self.assertEqual(RF.data_type, ('RF', 'single', [1, 2]), \
                         'RF data_type identifier incorrect')
        
        RF = dtypes.RF_section_function(1, [2, 3, 4, 5], harmonics = [1, 2])
        self.assertEqual(RF.data_type, ('RF', 'by_turn', [1, 2]), \
                         'RF data_type identifier incorrect')
        
        RF = dtypes.RF_section_function(1, inArr, harmonics = [1, 2])
        self.assertEqual(RF.data_type, ('RF', 'by_time', [1, 2]), \
                         'RF data_type identifier incorrect')
        
    
    def test_RF_intepolation(self):
        
        inArr1 = np.array([[1, 2, 3], [4, 5, 6]])
        inArr2 = np.array([[2, 4], [5, 6]])
        inArr3 = np.array([[0, 4, 5], [1, 1, 1]])
        
        RF = dtypes.RF_section_function(inArr1, inArr2, inArr3, 1, \
                                        harmonics=[1, 2, 3, 4], \
                                        interpolation='linear')
        
        self.assertEqual(RF[0][0].tolist(), [0, 1, 2, 3, 4, 5], \
                         'interpolation wrong time base')
        
        self.assertEqual(RF[0][0].tolist(), RF[1][0].tolist(), \
                         'interpolation wrong time base')
        self.assertEqual(RF[1][0].tolist(), RF[2][0].tolist(), \
                         'interpolation wrong time base')
        self.assertEqual(RF[2][0].tolist(), RF[3][0].tolist(), \
                         'interpolation wrong time base')
        
        self.assertEqual(RF.shape, (4, 2, 6), 'After interpolation shape wrong')
        
        with self.assertRaises(RuntimeError, msg = 'Non-linear interpolation '
                                               + 'should raise RuntimeError'):
            RF = dtypes.RF_section_function(inArr1, inArr2, inArr3, 1, \
                                        harmonics=[1, 2, 3, 4], \
                                        interpolation='cubic')


    def test_RF_exceptions(self):
        
        with self.assertRaises(exceptions.InputError, \
                               msg='If number of data points does not match ' \
                                + 'number of harmonics InputError should ' \
                                + 'be raised'):
            
            dtypes.RF_section_function(1, harmonics = [1, 2])
        
        with self.assertRaises(exceptions.DataDefinitionError, \
                               msg='Attempting interpolation with by_turn ' \
                               + 'data should raise DataDefinitionError'):
            
            dtypes.RF_section_function([1, 2, 3], [1, 2, 3], \
                                       harmonics = [1, 2], interpolation=True)


    def test__check_turn_numbers(self):

        dtypes._check_turn_numbers(([1, 2, 3], [1, 2, 3]), ('by_turn',)*2)
        lengths = dtypes._check_turn_numbers(([1, 2, 3], 1), \
                                             ('by_turn', 'single'), True)
        
        with self.assertRaises(exceptions.InputError, \
                               msg='by_turn mixed with single should throw ' \
                               + 'InputError when not explicitly allowed'):
            dtypes._check_turn_numbers(([1, 2, 3], 1), ('by_turn', 'single'))
        
        with self.assertRaises(exceptions.DataDefinitionError, \
                               msg='mixed number of turns should throw ' \
                               + 'DataDefinitionError'):
            dtypes._check_turn_numbers(([1, 2, 3], [1, 2]), ('by_turn',)*2)
        
        self.assertEqual(lengths, 3, msg='return number of turns wrong')


    def test__check_data_types(self):
        
        dtypes._check_data_types([1, 1, 1])
        dtypes._check_data_types([1, 1, 'single'], allow_single=True)
        
        with self.assertRaises(exceptions.DataDefinitionError, \
                               msg='mixed single and non-single data_type' \
                               + ' should throw DataDefinitionError unless ' \
                               + 'explicitly allowed'):
            dtypes._check_data_types([1, 1, 'single'])
            
        
    def test__check_time_turns(self):
        
        dtypes._check_time_turns(None, None)
        dtypes._check_time_turns(1, None)
        dtypes._check_time_turns(None, 1)
        
        with self.assertRaises(exceptions.InputError, msg='turns and times ' \
                               + 'both not None should raise InputError'):
            dtypes._check_time_turns(1, 1)
        

    def test__check_dims(self):
        
        dtypes._check_dims([1, 2, 3], n_turns = 3)
        
        inArr1 = np.array([[1, 2, 3], [4, 5, 6]])
        outArr, outType = dtypes._check_dims(inArr1)
        self.assertEqual(outArr.tolist(), inArr1.tolist(),\
                         'array has been altered')
        self.assertEqual(outType, 'by_time', 'type has been misidentified')
        
        outArr, outType = dtypes._check_dims(inArr1[0])
        self.assertEqual(outArr.tolist(), inArr1[0].tolist(),\
                         'array has been altered')
        self.assertEqual(outType, 'by_turn', 'type has been misidentified')
        
        outArr, outType = dtypes._check_dims(inArr1[0][0])
        self.assertEqual(outArr.tolist(), inArr1[0][0].tolist(),\
                         'array has been altered')
        self.assertEqual(outType, 'single', 'type has been misidentified')

        outArr, outType = dtypes._check_dims(inArr1[0][0], n_turns = 5)
        self.assertEqual(outArr, [inArr1[0][0]]*5,\
                         'array has been altered')
        self.assertEqual(outType, 'by_turn', 'type has been misidentified')


        with self.assertRaises(exceptions.InputError,
                               msg='Wrong number of turns should raise '\
                               + 'InputError'):
            outArr, outType = dtypes._check_dims(inArr1[0], n_turns = 4)

        with self.assertRaises(exceptions.InputError,
                               msg='If data has [time, data] format passing '\
                               + 'time as well should raise InputError'):
            outArr, outType = dtypes._check_dims(inArr1, time = [1, 2])

        with self.assertRaises(exceptions.InputError,
                               msg='Misshapped data should raise InputError'):
            outArr, outType = dtypes._check_dims(inArr1[0], inArr1)


if __name__ == '__main__':

    
    unittest.main()