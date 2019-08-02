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

thisDir = os.getcwd()
print(thisDir)
print(os.path.abspath(thisDir))
this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
#print(thisDir)
#if thisDir+'/../../' not in sys.path:
#    sys.path.insert(0, thisDir + '/../../')

#BLonD_Common imports
#----------------
if os.path.abspath(this_directory + '../../') not in sys.path:
    sys.path.insert(0, os.path.abspath(this_directory + '../../'))
#import data_common.datatypes.datatypes as dtypes
#from datatypes import datatypes as dtypes
import datatypes.datatypes as dtypes
import data_common.utilities.Exceptions as exceptions
print(exceptions)
class test_datatypes(unittest.TestCase):
    
    def test__function(self):
        
        with self.assertRaises(exceptions.InputError, \
                               msg='Undeclared data_type should raise InputError'):
            dtypes._function(1)
        
        test1 = np.array([[1, 2, 3], [4, 5, 6]])
        test2 = np.array([[1, 2], [4, 5]])
        
        with self.assertRaises(exceptions.InputError, \
                               msg='Misshaped input should raise InputError'):
            dtypes._function([test1, test2], 'test')

        



if __name__ == '__main__':

    print(exceptions.InputError)
    
    try:
        dtypes._function(1)
    except exceptions.InputError:
        print("test")
    
    unittest.main()