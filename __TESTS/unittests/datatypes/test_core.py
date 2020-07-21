# coding: utf8
# Copyright 2019 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Test dataypes._core.py

'''

# General imports
# ---------------
import sys
import unittest
import numpy as np
import numpy.testing as npTest
import os

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

# BLonD_Common imports
# --------------------
if os.path.abspath(this_directory + '../../../../') not in sys.path:
    sys.path.insert(0, os.path.abspath(this_directory + '../../../../'))

import blond_common.datatypes._core as core
from blond_common.devtools import exceptions

class test_core(unittest.TestCase):

    def test_creation(self):

        core._function(np.array([1, 2, 3]), data_type = {},
                                       interpolation = 'linear')
        core._function(np.array([[1, 2, 3], [2, 3, 4]]), data_type = {})
        core._function(np.zeros([3, 2, 10]), data_type = {})
        core._function(np.array([[1, 2, 3], [2, 3]]), data_type = {})


    def test_zeros(self):

        test = core._function.zeros([3, 3])
        self.assertEqual(len(test.shape), 2,
                                 'length of test.shape should be 2')
        self.assertEqual(test.shape[0], 3,
                                 'first element of shape should be 3')
        self.assertEqual(test.shape[1], 3,
                                 'second element of shape should be 3')

        self.assertIsNone(test.timebase, msg='timebase should be None')

        test = core._function.zeros([3, 3], data_type = {'timebase': 'test'})
        self.assertEqual(test.timebase, 'test',
                                 msg='timebase has not been correctly set')

        with self.assertRaises(exceptions.InputDataError, \
                               msg='Passing an invalid parameter should ' \
                               + 'raise an InputDataError'):

            test = core._function.zeros([3, 3], data_type = {'fake_parameter':
                                                                 'test'})

    def test_slicing(self):

        test = core._function.zeros(10, data_type={'timebase': 'fake'})
        sliced = test[:5]

        self.assertEqual(sliced.data_type['timebase'], 'fake',
                             msg='sliced data_type not copied correctly')
        self.assertIsInstance(sliced, core._function,
                              msg='Type incorrect after slicing')


    def test_copy(self):

        test = core._function.zeros(10, data_type={'timebase': 'fake'})
        copied = test.copy()

        self.assertEqual(copied.data_type['timebase'], 'fake',
                             msg='copied data_type not copied correctly')
        self.assertIsInstance(copied, core._function,
                                      msg='Type incorrect after copying')


    def test_reshape_basic(self):

        ############
        #TIME BASED#
        ############
        test = core._function(np.array([[[1, 2, 3], [1, 2, 3]]]),
                           data_type={'timebase': 'by_time'},
                           interpolation='linear')
        test = test.reshape(1, [1.5], [1])
        self.assertEqual(test, 1.5,
                                 msg='Interpolation not computed correctly')
        self.assertIsInstance(test, core._function,
                                  msg='Type incorrect after reshape')
        self.assertEqual(test.timebase, 'interpolated',
                         msg = 'After reshape timebase should be interpolated')

        test = core._function(np.array([[[1, 2, 3], [1, 2, 3]],
                                         [[1, 2, 3], [4, 5, 6]]]),
                           data_type={'timebase': 'by_time'},
                           interpolation='linear')
        test = test.reshape(2, [1.5], [1])

        self.assertEqual(test[0][0], 1.5,
                                 msg='Interpolation not computed correctly')
        self.assertEqual(test[1][0], 4.5,
                                 msg='Interpolation not computed correctly')
        self.assertIsInstance(test, core._function,
                                  msg='Type incorrect after reshape')
        self.assertEqual(test.timebase, 'interpolated',
                         msg = 'After reshape timebase should be interpolated')

        ############
        #TURN BASED#
        ############
        test = core._function(np.array([[1, 2, 3, 1, 2, 3]]),
                           data_type={'timebase': 'by_turn'},
                           interpolation='linear')
        test = test.reshape(1, [1.5], use_turns = [1])
        self.assertEqual(test, 2, msg='Interpolation not computed correctly')
        self.assertIsInstance(test, core._function,
                                  msg='Type incorrect after reshape')
        self.assertEqual(test.timebase, 'interpolated',
                         msg = 'After reshape timebase should be interpolated')

        test = core._function(np.array([[1, 2, 3, 1, 2, 3],
                                          [1, 2, 3, 4, 5, 6]]),
                            data_type={'timebase': 'by_turn'},
                            interpolation='linear')
        test = test.reshape(2, use_time = [1.5, 2.5], use_turns = [1, 3])

        self.assertEqual(test[0][0], 2,
                                  msg='Interpolation not computed correctly')
        self.assertEqual(test[0][1], 1,
                                  msg='Interpolation not computed correctly')
        self.assertEqual(test[1][0], 2,
                                  msg='Interpolation not computed correctly')
        self.assertEqual(test[1][1], 4,
                                  msg='Interpolation not computed correctly')
        self.assertIsInstance(test, core._function,
                                  msg='Type incorrect after reshape')
        self.assertEqual(test.timebase, 'interpolated',
                         msg = 'After reshape timebase should be interpolated')

        ###############
        #SINGLE VALUED#
        ###############
        test = core._function(np.array([1]),
                           data_type={'timebase': 'single'},
                           interpolation='linear')
        test = test.reshape(1, [1.5], use_turns = [1])
        self.assertEqual(test, 1, msg='Interpolation not computed correctly')
        self.assertIsInstance(test, core._function,
                                  msg='Type incorrect after reshape')
        self.assertEqual(test.timebase, 'interpolated',
                         msg = 'After reshape timebase should be interpolated')

        test = core._function(np.array([1, 2, 3]),
                            data_type={'timebase': 'single'},
                            interpolation='linear')
        test = test.reshape(3, use_time = [1.5, 2.5], use_turns = [1, 3])

        self.assertEqual(test[0][0], 1,
                                  msg='Interpolation not computed correctly')
        self.assertEqual(test[0][1], 1,
                                  msg='Interpolation not computed correctly')
        self.assertEqual(test[1][0], 2,
                                  msg='Interpolation not computed correctly')
        self.assertEqual(test[1][1], 2,
                                  msg='Interpolation not computed correctly')
        self.assertEqual(test[2][0], 3,
                                  msg='Interpolation not computed correctly')
        self.assertEqual(test[2][1], 3,
                                  msg='Interpolation not computed correctly')
        self.assertIsInstance(test, core._function,
                                  msg='Type incorrect after reshape')
        self.assertEqual(test.timebase, 'interpolated',
                         msg = 'After reshape timebase should be interpolated')


    def test_reshape_store_time(self):

        test = core._function(np.array([[[1, 2, 3], [1, 2, 3]],
                                          [[1, 2, 3], [4, 5, 6]]]),
                            data_type={'timebase': 'by_time'},
                            interpolation='linear')
        test = test.reshape(2, [1.5], store_time = True)

        self.assertEqual(test.shape, (2, 2, 1),
                             msg = 'The reshaped array has the wrong shape')
        self.assertEqual(test[0,0,0], 1.5,
                                     msg = 'The new time axis is incorrect')
        self.assertEqual(test[1,0,0], 1.5,
                                     msg = 'The new time axis is incorrect')
        self.assertEqual(test[0,1,0], 1.5,
                                     msg = 'The interpolation is incorrect')
        self.assertEqual(test[1,1,0], 4.5,
                                     msg = 'The interpolation is incorrect')

        test = core._function(np.array([1, 2]),
                              data_type={'timebase': 'single'},
                              interpolation='linear')
        test = test.reshape(2, [1.5], store_time = True)

        self.assertEqual(test.shape, (2, 2, 1),
                             msg = 'The reshaped array has the wrong shape')
        self.assertEqual(test[0,0,0], 1.5,
                                     msg = 'The new time axis is incorrect')
        self.assertEqual(test[1,0,0], 1.5,
                                     msg = 'The new time axis is incorrect')
        self.assertEqual(test[0,1,0], 1,
                                     msg = 'The interpolation is incorrect')
        self.assertEqual(test[1,1,0], 2,
                                     msg = 'The interpolation is incorrect')

        test = core._function.zeros([2, 3], data_type={'timebase':
                                                           'by_turn'})
        with self.assertRaises(exceptions.InputError, \
                               msg='store_time should raise an InputError '
                                   +'for a function defined by_turn'):
            test.reshape(2, use_turns = [1], store_time = True)


    def test_multiplication(self):

        test = core._function(2, data_type = {'timebase': 'single'},
                              interpolation = 'linear')
        test *= 5
        self.assertEqual(test, 10, msg='Single value in place multiplication '
                                         + 'incorrect')

        test = core._function(2, data_type = {'timebase': 'single'},
                              interpolation = 'linear')
        test2 = test * 2
        self.assertEqual(test2, 4, msg='Single value multiplication '
                                         + 'incorrect')
        test2 = 2 * test
        self.assertEqual(test2, 4, msg='Single value rmultiplication '
                                         + 'incorrect')

        test = core._function([[1, 2, 3], [4, 5, 6]],
                              data_type = {'timebase': 'by_turn'},
                              interpolation = 'linear')
        test *= 2
        compare = [[2, 4, 6], [8, 10, 12]]
        npTest.assert_array_equal(test, compare,
                              err_msg = "by turn in place multiplication " +
                                          "incorrect")

        test = core._function([[1, 2, 3], [4, 5, 6]],
                              data_type = {'timebase': 'by_turn'},
                              interpolation = 'linear')
        test2 = test * 2
        compare = [[2, 4, 6], [8, 10, 12]]
        npTest.assert_array_equal(test2, compare,
                              err_msg = "by turn multiplication incorrect")
        test2 = 2 * test
        compare = [[2, 4, 6], [8, 10, 12]]
        npTest.assert_array_equal(test2, compare,
                              err_msg = "by turn rmultiplication incorrect")

        test = core._function([[[1, 2, 3], [4, 5, 6]]],
                              data_type = {'timebase': 'by_time'},
                              interpolation = 'linear')
        test *= 2
        compare = [[[1, 2, 3], [8, 10, 12]]]
        npTest.assert_array_equal(test, compare,
                              err_msg = "by time in place multiplication "
                                          + "incorrect")

        test = core._function([[[1, 2, 3], [4, 5, 6]]],
                              data_type = {'timebase': 'by_time'},
                              interpolation = 'linear')
        test2 = test * 2
        compare = [[[1, 2, 3], [8, 10, 12]]]
        npTest.assert_array_equal(test2, compare,
                              err_msg = "by time multiplication incorrect")
        test2 = 2 * test
        compare = [[[1, 2, 3], [8, 10, 12]]]
        npTest.assert_array_equal(test2, compare,
                              err_msg = "by time rmultiplication incorrect")


    def test_addition(self):

        test = core._function(1, data_type = {'timebase': 'single'},
                              interpolation = 'linear')
        test += 5
        self.assertEqual(test, 6, msg='Single value in place addition '
                                         + 'incorrect')

        test = core._function(1, data_type = {'timebase': 'single'},
                              interpolation = 'linear')
        test2 = test + 1
        self.assertEqual(test2, 2, msg='Single value addition '
                                         + 'incorrect')
        test2 = 1 + test
        self.assertEqual(test2, 2, msg='Single value raddition '
                                         + 'incorrect')

        test = core._function([[1, 2, 3], [4, 5, 6]],
                              data_type = {'timebase': 'by_turn'},
                              interpolation = 'linear')
        test += 5
        compare = [[6, 7, 8], [9, 10, 11]]
        npTest.assert_array_equal(test, compare,
                              err_msg = "by turn in place addition incorrect")

        test = core._function([[1, 2, 3], [4, 5, 6]],
                              data_type = {'timebase': 'by_turn'},
                              interpolation = 'linear')
        test2 = test + 1
        compare = [[2, 3, 4], [5, 6, 7]]
        npTest.assert_array_equal(test2, compare,
                              err_msg = "by turn addition incorrect")
        test2 = 1 + test
        compare = [[2, 3, 4], [5, 6, 7]]
        npTest.assert_array_equal(test2, compare,
                              err_msg = "by turn raddition incorrect")

        test = core._function([[[1, 2, 3], [4, 5, 6]]],
                              data_type = {'timebase': 'by_time'},
                              interpolation = 'linear')
        test += 5
        compare = [[[1, 2, 3], [9, 10, 11]]]
        npTest.assert_array_equal(test, compare,
                              err_msg = "by time in place addition incorrect")

        test = core._function([[[1, 2, 3], [4, 5, 6]]],
                              data_type = {'timebase': 'by_time'},
                              interpolation = 'linear')
        test2 = test + 1
        compare = [[[1, 2, 3], [5, 6, 7]]]
        npTest.assert_array_equal(test2, compare,
                              err_msg = "by time addition incorrect")
        test2 = 1 + test
        compare = [[[1, 2, 3], [5, 6, 7]]]
        npTest.assert_array_equal(test2, compare,
                              err_msg = "by time raddition incorrect")


    def test_subtraction(self):

        test = core._function(1, data_type = {'timebase': 'single'},
                              interpolation = 'linear')
        test -= 5
        self.assertEqual(test, -4, msg='Single value in place subtraction '
                                         + 'incorrect')

        test = core._function(1, data_type = {'timebase': 'single'},
                              interpolation = 'linear')
        test2 = test - 1
        self.assertEqual(test2, 0, msg='Single value subtraction '
                                         + 'incorrect')

        test = core._function([[1, 2, 3], [4, 5, 6]],
                              data_type = {'timebase': 'by_turn'},
                              interpolation = 'linear')
        test -= 5
        compare = [[-4, -3, -2], [-1, 0, 1]]
        npTest.assert_array_equal(test, compare,
                              err_msg = "by turn in place subtraction incorrect")

        test = core._function([[1, 2, 3], [4, 5, 6]],
                              data_type = {'timebase': 'by_turn'},
                              interpolation = 'linear')
        test2 = test - 1
        compare = [[0, 1, 2], [3, 4, 5]]
        npTest.assert_array_equal(test2, compare,
                              err_msg = "by turn subtraction incorrect")

        test = core._function([[[1, 2, 3], [4, 5, 6]]],
                              data_type = {'timebase': 'by_time'},
                              interpolation = 'linear')
        test -= 5
        compare = [[[1, 2, 3], [-1, 0, 1]]]
        npTest.assert_array_equal(test, compare,
                              err_msg = "by time in place subtraction incorrect")

        test = core._function([[[1, 2, 3], [4, 5, 6]]],
                              data_type = {'timebase': 'by_time'},
                              interpolation = 'linear')
        test2 = test - 1
        compare = [[[1, 2, 3], [3, 4, 5]]]
        npTest.assert_array_equal(test2, compare,
                              err_msg = "by time subtraction incorrect")


    def test_exceptions(self):

        test1 = core._function([[[1, 2, 3], [4, 5, 6]]],
                              data_type = {'timebase': 'by_time'},
                              interpolation = 'linear')

        test2 = core._function([[1, 2, 3]],
                              data_type = {'timebase': 'by_turn'},
                              interpolation = 'linear')

        with self.assertRaises(TypeError, msg='addition with different '
                               + 'data_type dicts should raise a TypeError'):
            test1 += test2
        with self.assertRaises(TypeError, msg='subtraction with different '
                               + 'data_type dicts should raise a TypeError'):
            test1 -= test2
        with self.assertRaises(TypeError, msg='multiplication with different '
                               + 'data_type dicts should raise a TypeError'):
            test1 *= test2
        with self.assertRaises(TypeError, msg='division with different '
                               + 'data_type dicts should raise a TypeError'):
            test1 /= test2

        with self.assertRaises(exceptions.InputDataError, msg='unrecognised '
                               +'data_type dict options should raise an '
                               +'InputDataError.'):
            core._function(1, data_type = {'fake_option_string': None},
                              interpolation = 'linear')

        with self.assertRaises(exceptions.InputError, msg='_prep_reshape '
                               +'with use_time=None and use_turns=None should '
                               'raise an InputError.'):
            test1._prep_reshape(1)

        with self.assertRaises(exceptions.InputError, msg='_prep_reshape '
                               +'with use_turns should raise an InputError if'
                               +' timebase == "by_time"'):
            test1._prep_reshape(1, use_turns=[1, 2])

        with self.assertRaises(exceptions.InputError, msg='_prep_reshape '
                               +'with use_time should raise an InputError if'
                               +' timebase == "by_turn"'):
            test2._prep_reshape(1, use_time=[1, 2])


if __name__ == '__main__':

    unittest.main()