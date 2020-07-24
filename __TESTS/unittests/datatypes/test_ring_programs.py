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
import scipy.constants as cont

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

# BLonD_Common imports
# --------------------
if os.path.abspath(this_directory + '../../../../') not in sys.path:
    sys.path.insert(0, os.path.abspath(this_directory + '../../../../'))

import blond_common.datatypes.ring_programs as ringProg
from blond_common.devtools import exceptions

class test_ring_programs(unittest.TestCase):

    def test_basic_ring_function_creation(self):

        func = ringProg._ring_function(1E9)
        expectDict = {'timebase': 'single', 'sectioning': 'single_section',
                      'interpolation': None}
        self.assertEqual(func.shape, (1,), 'Single value shape incorrect')
        self.assertEqual(func.data_type, expectDict, 'data_type dict contents'
                                                     +' incorrect')

        func = ringProg._ring_function([1E9, 1E9], [1E9, 1E9])
        expectDict = {'timebase': 'by_turn', 'sectioning': 'multi_section',
                      'interpolation': None}
        self.assertEqual(func.shape, (2, 2), 'Multi-section by turn shape '
                                           + 'incorrect')
        self.assertEqual(func.data_type, expectDict, 'data_type dict contents'
                                                     +' incorrect')

        func = ringProg._ring_function([1E9, 1E9], time = [0, 1])
        expectDict = {'timebase': 'by_time', 'sectioning': 'single_section',
                      'interpolation': None}
        self.assertEqual(func.shape, (1, 2, 2), 'Single-section by time shape '
                                              + 'incorrect')
        self.assertEqual(func.data_type, expectDict, 'data_type dict contents'
                                                     +' incorrect')

        func = ringProg._ring_function([[0, 1], [1E9, 1E9]],
                                        [[0, 1], [1E9, 1E9]])
        expectDict = {'timebase': 'by_time', 'sectioning': 'multi_section',
                      'interpolation': None}
        self.assertEqual(func.shape, (2, 2, 2), 'Multi-section by time shape '
                                              + 'incorrect')
        self.assertEqual(func.data_type, expectDict, 'data_type dict contents'
                                                      +' incorrect')

        func = ringProg._ring_function(1, n_turns = 100)
        expectDict = {'timebase': 'by_turn', 'sectioning': 'single_section',
                      'interpolation': None}
        self.assertEqual(func.shape, (1, 100), 'single section by turn '
                                              +'expanded shape incorrect')
        self.assertEqual(func.data_type, expectDict, 'data_type dict contents'
                                                      +' incorrect')

        func = ringProg._ring_function(1, time = [0, 1])
        expectDict = {'timebase': 'by_time', 'sectioning': 'single_section',
                      'interpolation': None}
        self.assertEqual(func.shape, (1, 2, 2), 'single section by turn '
                                              +'expanded shape incorrect')
        self.assertEqual(func.data_type, expectDict, 'data_type dict contents'
                                                      +' incorrect')

        func = ringProg._ring_function(1, 2, time = [0, 1])
        expectDict = {'timebase': 'by_time', 'sectioning': 'multi_section',
                      'interpolation': None}
        self.assertEqual(func.shape, (2, 2, 2), 'multi section by turn '
                                              +'expanded shape incorrect')
        self.assertEqual(func.data_type, expectDict, 'data_type dict contents'
                                                      +' incorrect')


    def test_mixed_input(self):

        with self.assertRaises(exceptions.DataDefinitionError, \
                               msg='Declaring a mix of turn based and single'
                                 + ' valued input should raise a '
                                 + 'DataDefinitionError with default of '
                                 + 'allow_single = False'):
            func = ringProg._ring_function(1, [1, 2])

        func = ringProg._ring_function(1, [1, 2], allow_single = True)
        expectDict = {'timebase': 'by_turn', 'sectioning': 'multi_section',
                      'interpolation': None}
        self.assertEqual(func.shape, (2, 2), 'multi section by turn '
                                              +'expanded shape incorrect')
        self.assertEqual(func.data_type, expectDict, 'data_type dict contents'
                                                      +' incorrect')

        with self.assertRaises(exceptions.DataDefinitionError, \
                               msg='Declaring a mix of time based and single'
                                 + ' valued input should raise a '
                                 + 'DataDefinitionError with default of '
                                 + 'allow_single = False'):
            func = ringProg._ring_function(1, [[0, 1], [1, 2]])

        func = ringProg._ring_function(1, [[0, 1], [1, 2]], allow_single=True)
        expectDict = {'timebase': 'by_time', 'sectioning': 'multi_section',
                      'interpolation': None}
        self.assertEqual(func.shape, (2, 2, 2), 'multi section by time '
                                              +'expanded shape incorrect')
        self.assertEqual(func.data_type, expectDict, 'data_type dict contents'
                                                      +' incorrect')


    def test_synchronous_data(self):

        mom = ringProg.momentum_program(1E9)
        totEn = ringProg.total_energy_program([2E9]*10)
        kinEn = ringProg.kinetic_energy_program([[0, 1], [160E6, 1E9]])
        bend = ringProg.bending_field_program(0.5, n_turns = 1000)

        self.assertEqual(mom.source, 'momentum', msg='momentum_program source'
                                                 + ' str incorrect')
        self.assertEqual(totEn.source, 'energy', msg='momentum_program source'
                                                 + ' str incorrect')
        self.assertEqual(kinEn.source, 'kin_energy', msg='momentum_program'
                                                 + ' source str incorrect')
        self.assertEqual(bend.source, 'B_field', msg='momentum_program source'
                                                 + ' str incorrect')

        conversions = {'momentum': ringProg.momentum_program,
                       'energy': ringProg.total_energy_program,
                       'kin_energy': ringProg.kinetic_energy_program,
                       'B_field': ringProg.bending_field_program}

        self.assertEqual(mom._conversions, conversions,
                         msg = 'Momentum program _conversions dict incorrect')
        self.assertEqual(totEn._conversions, conversions,
                         msg = 'Energy program _conversions dict incorrect')
        self.assertEqual(kinEn._conversions, conversions,
                         msg = 'KinEnergy program _conversions dict incorrect')
        self.assertEqual(bend._conversions, conversions,
                         msg = 'B Field program _conversions dict incorrect')

        m_p = cont.physical_constants['proton mass energy equivalent in MeV']
        m_p = m_p[0]*1E6

        #TODO: Check values and permutations
        ringProg.momentum_program.combine_single_sections(mom, mom)
        ringProg.momentum_program.combine_single_sections(totEn, totEn,
                                                          rest_mass = m_p)
        ringProg.momentum_program.combine_single_sections(kinEn, kinEn,
                                                          rest_mass = m_p,
                                                        interpolation='linear')
        ringProg.momentum_program.combine_single_sections(bend, bend,
                                                          charge = 1,
                                                          bending_radius = 8)


    def test_synchronous_conversions(self):

        m_p = cont.physical_constants['proton mass energy equivalent in MeV']
        m_p = m_p[0]*1E6
        charge = 1
        rho = 8.7

        ###############
        #FROM MOMENTUM#
        ###############
        mom = ringProg.momentum_program(1E9)

        newB = mom.to_B_field(False, bending_radius = rho, charge = charge)
        self.assertIsInstance(newB, ringProg.bending_field_program,
                              msg = 'type cast to bending_field_program '
                                  + 'incorrect')
        self.assertAlmostEqual(newB[0], 0.38340701, places = 4,
                               msg='P to B conversion inaccurate')
        self.assertEqual(newB.data_type, mom.data_type,
                         msg = 'data_type dict not copied correctly')

        newE = mom.to_total_energy(False, rest_mass = m_p)
        self.assertIsInstance(newE, ringProg.total_energy_program,
                              msg = 'type cast to total_energy_program '
                                  + 'incorrect')
        self.assertAlmostEqual(newE[0]/1E9, 1.37126019, places = 4,
                               msg='P to E conversion inaccurate')
        self.assertEqual(newE.data_type, mom.data_type,
                         msg = 'data_type dict not copied correctly')

        newKE = mom.to_kin_energy(False, rest_mass = m_p)
        self.assertIsInstance(newKE, ringProg.kinetic_energy_program,
                              msg = 'type cast to kinetic_energy_program '
                                  + 'incorrect')
        self.assertAlmostEqual(newKE[0]/1E8, 4.32988103, places = 4,
                               msg='P to KE conversion inaccurate')
        self.assertEqual(newKE.data_type, mom.data_type,
                          msg = 'data_type dict not copied correctly')

        newP = mom.to_momentum(False)
        self.assertIsInstance(newP, ringProg.momentum_program,
                              msg = 'type cast to momentum_program '
                                  + 'incorrect')
        self.assertAlmostEqual(newP[0]/1E8, mom[0]/1E8, places = 9,
                               msg='P to P conversion inaccurate')
        self.assertEqual(newP.data_type, mom.data_type,
                          msg = 'data_type dict not copied correctly')


        ####################
        #FROM BENDING FIELD#
        ####################
        B = ringProg.bending_field_program([0.5, 0.5])

        newP = B.to_momentum(False, bending_radius = rho, charge = charge)
        self.assertIsInstance(newP, ringProg.momentum_program,
                              msg = 'type cast to momentum_program '
                                  + 'incorrect')
        np.testing.assert_array_almost_equal(newP/1E9, [[1.30409719]*2],
                                             decimal = 4, err_msg='B to P '
                                             + 'conversion inaccurate')
        self.assertEqual(newP.data_type, B.data_type,
                         msg = 'data_type dict not copied correctly')

        newE = B.to_total_energy(False, rest_mass = m_p, bending_radius = rho,
                                 charge = 1)
        self.assertIsInstance(newE, ringProg.total_energy_program,
                              msg = 'type cast to total_energy_program '
                                  + 'incorrect')
        np.testing.assert_array_almost_equal(newE/1E9, [[1.60655657]*2],
                                             decimal = 4, err_msg='B to E '
                                             + 'conversion inaccurate')
        self.assertEqual(newE.data_type, B.data_type,
                         msg = 'data_type dict not copied correctly')

        newKE = B.to_kin_energy(False, rest_mass = m_p, bending_radius = rho,
                                 charge = 1)
        self.assertIsInstance(newKE, ringProg.kinetic_energy_program,
                              msg = 'type cast to kinetic_energy_program '
                                  + 'incorrect')
        np.testing.assert_array_almost_equal(newKE/1E8, [[6.68284477]*2],
                                             decimal = 4, err_msg='B to KE '
                                             + 'conversion inaccurate')
        self.assertEqual(newKE.data_type, B.data_type,
                          msg = 'data_type dict not copied correctly')

        newB = B.to_B_field(False)
        self.assertIsInstance(newB, ringProg.bending_field_program,
                              msg = 'type cast to bending_field_program '
                                  + 'incorrect')
        np.testing.assert_array_almost_equal(newB, B,
                                             decimal = 9, err_msg='B to B '
                                             + 'conversion inaccurate')
        self.assertEqual(newB.data_type, B.data_type,
                          msg = 'data_type dict not copied correctly')


        ###################
        #FROM TOTAL ENERGY#
        ###################
        E = ringProg.total_energy_program(2E9, 2E9, time = [0, 1])

        compareArray = np.zeros([2, 2, 2])
        compareArray[:,0] = [0, 1]

        newP = E.to_momentum(False, rest_mass = m_p)
        compareArray[:,1] = 1.76625182
        self.assertIsInstance(newP, ringProg.momentum_program,
                              msg = 'type cast to momentum_program '
                                  + 'incorrect')
        np.testing.assert_array_almost_equal(newP/1E9, compareArray,
                                             decimal = 4, err_msg='E to P '
                                             + 'conversion inaccurate')
        self.assertEqual(newP.data_type, E.data_type,
                         msg = 'data_type dict not copied correctly')

        newB = E.to_B_field(False, rest_mass = m_p, bending_radius = rho,
                            charge = 1)
        compareArray[:,1] = 0.67719332
        self.assertIsInstance(newB, ringProg.bending_field_program,
                              msg = 'type cast to bending_field_program '
                                  + 'incorrect')
        np.testing.assert_array_almost_equal(newB, compareArray,
                                             decimal = 4, err_msg='E to B '
                                             + 'conversion inaccurate')
        self.assertEqual(newE.data_type, B.data_type,
                         msg = 'data_type dict not copied correctly')

        newKE = E.to_kin_energy(False, rest_mass = m_p)
        compareArray[:,1] = 1.06172791
        self.assertIsInstance(newKE, ringProg.kinetic_energy_program,
                              msg = 'type cast to kinetic_energy_program '
                                  + 'incorrect')
        np.testing.assert_array_almost_equal(newKE/1E9, compareArray,
                                             decimal = 4, err_msg='E to KE '
                                             + 'conversion inaccurate')
        self.assertEqual(newKE.data_type, E.data_type,
                          msg = 'data_type dict not copied correctly')

        newE = E.to_total_energy(False)
        compareArray[:,1] = E[:,1]
        self.assertIsInstance(newE, ringProg.total_energy_program,
                              msg = 'type cast to total_energy_program '
                                  + 'incorrect')
        np.testing.assert_array_almost_equal(newE, compareArray,
                                             decimal = 9, err_msg='E to E '
                                             + 'conversion inaccurate')
        self.assertEqual(newE.data_type, E.data_type,
                          msg = 'data_type dict not copied correctly')


        #####################
        #FROM KINETIC ENERGY#
        #####################
        KE = ringProg.kinetic_energy_program([160E6, 250E6], [160E6, 250E6])

        compareArray = np.zeros([2, 2])

        newP = KE.to_momentum(False, rest_mass = m_p)
        compareArray[:,0] = 5.70830157
        compareArray[:,1] = 7.29133763
        self.assertIsInstance(newP, ringProg.momentum_program,
                              msg = 'type cast to momentum_program '
                                  + 'incorrect')
        np.testing.assert_array_almost_equal(newP/1E8, compareArray,
                                             decimal = 4, err_msg='KE to P '
                                             + 'conversion inaccurate')
        self.assertEqual(newP.data_type, KE.data_type,
                         msg = 'data_type dict not copied correctly')

        newB = KE.to_B_field(False, rest_mass = m_p, bending_radius = rho,
                             charge = 1)
        compareArray[:,0] = 0.21886028
        compareArray[:,1] = 0.27955499
        self.assertIsInstance(newB, ringProg.bending_field_program,
                              msg = 'type cast to bending_field_program '
                                  + 'incorrect')
        np.testing.assert_array_almost_equal(newB, compareArray,
                                             decimal = 4, err_msg='KE to B '
                                             + 'conversion inaccurate')
        self.assertEqual(newB.data_type, KE.data_type,
                         msg = 'data_type dict not copied correctly')

        newE = KE.to_total_energy(False, rest_mass = m_p)
        compareArray[:,0] = 1.09827209
        compareArray[:,1] = 1.18827209
        self.assertIsInstance(newE, ringProg.total_energy_program,
                              msg = 'type cast to total_energy_program '
                                  + 'incorrect')
        np.testing.assert_array_almost_equal(newE/1E9, compareArray,
                                             decimal = 4, err_msg='KE to E '
                                             + 'conversion inaccurate')
        self.assertEqual(newE.data_type, KE.data_type,
                          msg = 'data_type dict not copied correctly')

        newKE = KE.to_kin_energy(False)
        compareArray[:,0] = KE[:,0]
        compareArray[:,1] = KE[:,1]
        self.assertIsInstance(newKE, ringProg.kinetic_energy_program,
                              msg = 'type cast to kinetic_energy_program '
                                  + 'incorrect')
        np.testing.assert_array_almost_equal(newKE, compareArray,
                                             decimal = 9, err_msg='KE to E '
                                             + 'conversion inaccurate')
        self.assertEqual(newKE.data_type, KE.data_type,
                          msg = 'data_type dict not copied correctly')


    def test_momentum_compaction(self):

        alpha0 = ringProg.momentum_compaction(1)
        expectDType = {'timebase': 'single', 'order': 0,
                       'interpolation': 'linear',
                       'sectioning': 'single_section'}
        self.assertEqual(alpha0.data_type, expectDType,
                         msg='Data type dictionary does not match expected')

        alpha1 = ringProg.momentum_compaction([1, 2], order=1)
        expectDType = {'timebase': 'by_turn', 'order': 1,
                       'interpolation': 'linear',
                       'sectioning': 'single_section'}
        self.assertEqual(alpha1.data_type, expectDType,
                         msg='Data type dictionary does not match expected')

        with self.assertRaises(exceptions.InputError,
                           msg='momentum_compaction.combine_single_sections '
                              +'should raise an InputError if different orders'
                              +' of alpha are given.'):
            ringProg.momentum_compaction.combine_single_sections(alpha0,
                                                                 alpha1)

        alpha0_2 = ringProg.momentum_compaction(1, [1, 2])
        expectDType = {'timebase': 'by_turn', 'order': 0,
                       'interpolation': 'linear',
                       'sectioning': 'multi_section'}
        self.assertEqual(alpha0_2.data_type, expectDType,
                         msg='Data type dictionary does not match expected')

        alpha1_2 = ringProg.momentum_compaction(1, time = [1, 2], order=1)
        expectDType = {'timebase': 'by_time', 'order': 1,
                       'interpolation': 'linear',
                       'sectioning': 'single_section'}
        self.assertEqual(alpha1_2.data_type, expectDType,
                         msg='Data type dictionary does not match expected')

        with self.assertRaises(exceptions.InputError,
                           msg='momentum_compaction.combine_single_sections '
                              +'should raise an InputError if different '
                              +'timebases are used.'):
            ringProg.momentum_compaction.combine_single_sections(alpha1,
                                                                 alpha1_2)

if __name__ == '__main__':

    unittest.main()