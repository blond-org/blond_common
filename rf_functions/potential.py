# coding: utf8
# Copyright 2014-2019 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Module containing base functions to compute like rf voltage, rf potential,
separatrices, acceptance, emittance, synchrotron frequency**

:Authors: **Simon Albright**, **Alexandre Lasheen**
'''

# External imports
import numpy as np
import scipy.interpolate as interp
from ..maths.calculus import integ_cubic, deriv_cubic, minmax_location_cubic

# BLonD_Common imports
from ..devtools import exceptions as excpt


def rf_voltage_generation(n_points, t_rev, voltage, harmonic_number,
                          phi_offset, time_bounds=None):

    voltage = np.array(voltage, ndmin=1)
    harmonic_number = np.array(harmonic_number, ndmin=1)
    phi_offset = np.array(phi_offset, ndmin=1)

    if time_bounds is None:
        left_time = 0
        right_time = t_rev / harmonic_number[0]
        margin = 0.2
    else:
        left_time = time_bounds[0]
        right_time = time_bounds[1]
        margin = 0

    omega_rev = 2*np.pi/t_rev

    time_array = np.linspace(left_time-left_time*margin,
                             right_time+right_time*margin,
                             n_points)

    voltage_array = np.zeros(len(time_array))

    for indexRF in range(len(voltage)):
        voltage_array += voltage[indexRF] * np.sin(
            harmonic_number[indexRF]*omega_rev*time_array+phi_offset[indexRF])

    return time_array, voltage_array


def rf_potential_generation(n_points, t_rev, voltage, harmonic_number,
                            phi_offset, eta_0, charge, energy_increment,
                            time_bounds=None):

    voltage = np.array(voltage, ndmin=1)
    harmonic_number = np.array(harmonic_number, ndmin=1)
    phi_offset = np.array(phi_offset, ndmin=1)

    if time_bounds is None:
        left_time = 0
        right_time = t_rev / harmonic_number[0]
        margin = 0.2
    else:
        left_time = time_bounds[0]
        right_time = time_bounds[1]
        margin = 0

    omega_rev = 2*np.pi/t_rev

    time_array = np.linspace(left_time-left_time*margin,
                             right_time+right_time*margin,
                             n_points)

    eom_factor_potential = np.sign(eta_0) * charge / t_rev

    potential_well = eom_factor_potential*energy_increment/abs(charge) * \
        time_array

    for indexRF in range(len(voltage)):
        potential_well += eom_factor_potential * \
            voltage[indexRF]/(harmonic_number[indexRF]*omega_rev) * np.cos(
                harmonic_number[indexRF]*omega_rev*time_array +
                phi_offset[indexRF])

    return time_array, potential_well


def rf_potential_generation_cubic(time_array, voltage_array, eta_0, charge,
                                  t_rev, energy_increment,
                                  interpolated_voltage_minus_increment=None):

    eom_factor_potential = np.sign(eta_0) * charge / t_rev

    if interpolated_voltage_minus_increment is None:
        voltage_minus_increment = voltage_array - \
            (energy_increment)/abs(charge)
        interpolated_voltage_minus_increment = interp.splrep(
            time_array, voltage_minus_increment)
    else:
        pass

    potential_well = - eom_factor_potential * integ_cubic(
        time_array, voltage_minus_increment,
        tck=interpolated_voltage_minus_increment)[1]

    return time_array, potential_well, (voltage_minus_increment,
                                        interpolated_voltage_minus_increment)


# Defining a routine to locate potential wells and inner separatrices
def find_potential_wells_cubic(time_array_full, potential_well_full,
                               relative_max_val_precision_limit=1e-6,
                               mest=10, edge_is_max=False,
                               verbose=False):

    potwell_max_locs = []
    potwell_max_vals = []
    potwell_inner_max = []
    potwell_min_locs = []
    potwell_min_vals = []

    time_resolution = time_array_full[1]-time_array_full[0]

    tck = interp.splrep(time_array_full, potential_well_full)

    min_max_results = minmax_location_cubic(time_array_full,
                                            potential_well_full,
                                            tck=tck,
                                            mest=mest)

    min_pos = min_max_results[0][0]
    max_pos = min_max_results[0][1]
    min_val = min_max_results[1][0]
    max_val = min_max_results[1][1]

    most_left_min_pos = np.min(min_pos)
    most_right_min_pos = np.max(min_pos)
    most_left_max_pos = np.min(max_pos)
    most_right_max_pos = np.max(max_pos)

    left_edge_is_max = False
    right_edge_is_max = False
    if edge_is_max:
        # Adding the left or right edge to the list of max
        # The addition is done only if there is a min between the closest
        # max to the left/right edge and the actual edge
        # (avoid consecutive maxes)
        if potential_well_full[0] > potential_well_full[-1]:
            if most_left_min_pos > most_left_max_pos:
                max_pos = np.insert(max_pos, 0, time_array_full[0])
                max_val = np.insert(max_val, 0, potential_well_full[0])
                left_edge_is_max = True
        else:
            if most_right_min_pos < most_right_max_pos:
                max_pos = np.append(max_pos, time_array_full[-1])
                max_val = np.append(max_val, potential_well_full[-1])
                right_edge_is_max = True

    for index_max in range(len(max_val)):

        # Setting a max
        present_max_val = max_val[index_max]
        present_max_pos = max_pos[index_max]

        # Resetting the inner separatrix flag to 0
        inner_sep_max_left = np.nan
        inner_sep_max_right = np.nan

        # Checking left
        # This is a right max, checking for the left counterparts

        for index_left in range(index_max+2):
            if left_edge_is_max and (index_max == 0):
                # The left edge was manually added as a maximum, no check
                # to the left
                break

            if (index_left == 0) and (index_max == 0):
                # This is the most left max
                if most_left_min_pos > max_pos[index_max]:
                    break
                else:
                    left_max_val = potential_well_full[0]
                    left_max_pos = time_array_full[0]
            elif (index_left == 1) and (index_max == 0):
                # This indexes set is there to avoid checking the left edge
                # twice while one most left max
                continue
            elif (index_left == 0) and (index_max != 0):
                # This indexes set corresponds to the same max
                continue

            elif (index_left == (index_max+1)) and (index_max != 0):

                # No more max on the left, checking edge
                left_max_val = potential_well_full[0]
                left_max_pos = time_array_full[0]
            else:
                left_max_val = max_val[index_max-index_left]
                left_max_pos = max_pos[index_max-index_left]

            right_pos = present_max_pos
            right_val = present_max_val

            if np.isclose(left_max_val, present_max_val,
                          rtol=relative_max_val_precision_limit, atol=0):
                # The left max is identical to the present max, a pot. well
                # is found
                left_pos = left_max_pos
                left_val = left_max_val

                if [left_pos, right_pos] not in potwell_max_locs:
                    potwell_max_locs.append([left_pos, right_pos])
                    potwell_max_vals.append([left_val, right_val])

                    if np.isnan(inner_sep_max_left):
                        potwell_inner_max.append(np.nan)
                        potwell_min_locs.append(
                                float(min_pos[(min_pos > left_pos) *
                                              (min_pos < right_pos)]))
                        potwell_min_vals.append(
                                float(min_val[(min_pos > left_pos) *
                                              (min_pos < right_pos)]))

                    else:
                        potwell_inner_max.append(
                                float(inner_sep_max_left))
                        potwell_min_locs.append(np.nan)
                        potwell_min_vals.append(np.nan)

                    if verbose:
                        print('+L1 - IMAX '+str(index_max)+' - ILEFT ' +
                              str(index_left))
                        print([left_pos, right_pos], inner_sep_max_left)
                else:
                    pass
                    if verbose:
                        print('=L1 - IMAX '+str(index_max)+' - ILEFT ' +
                              str(index_left))

                # Breaking the loop
                break

            elif left_max_val < present_max_val:
                # The left max is smaller than the present max, this
                # means that there is an inner separatrix

                inner_sep_max_left = np.nanmax([
                    inner_sep_max_left, left_max_val])

                if verbose:
                    print('L2 - IMAX '+str(index_max)+' - ILEFT ' +
                          str(index_left))
                    print (inner_sep_max_left)

            elif left_max_val > present_max_val:
                # The left max is higher than the present max, finding
                # the intersection and breaking the loop

                indexes_find_root = np.where(
                    (tck[0] >= (left_max_pos-3*time_resolution)) *
                    (tck[0] < present_max_pos))

                tck_adjusted = (
                    tck[0][indexes_find_root],
                    (tck[1]-present_max_val)[indexes_find_root],
                    tck[2])

                potential_well_roots = interp.sproot(tck_adjusted,
                                                     mest=mest)

                # Breaking if root finding fails (bucket too small or
                # precision param too fine)
                if len(potential_well_roots) == 0:
                    print('Warning: could not intersect potential well ' +
                          'on the left! ' +
                          'Try lowering relative_max_val_precision_limit')
                    break

                left_pos = np.max(potential_well_roots)
                left_val = present_max_val

                if [left_pos, right_pos] not in potwell_max_locs:
                    potwell_max_locs.append([left_pos, right_pos])
                    potwell_max_vals.append([left_val, right_val])

                    if np.isnan(inner_sep_max_left):
                        potwell_inner_max.append(np.nan)
                        potwell_min_locs.append(
                                float(min_pos[(min_pos > left_pos) *
                                              (min_pos < right_pos)]))
                        potwell_min_vals.append(
                                float(min_val[(min_pos > left_pos) *
                                              (min_pos < right_pos)]))

                    else:
                        potwell_inner_max.append(
                                float(inner_sep_max_left))
                        potwell_min_locs.append(np.nan)
                        potwell_min_vals.append(np.nan)

                    if verbose:
                        print('+L3 - IMAX '+str(index_max)+' - ILEFT ' +
                              str(index_left))
                        print([left_pos, right_pos], inner_sep_max_left)
                else:
                    pass
                    if verbose:
                        print('=L3 - IMAX '+str(index_max)+' - ILEFT ' +
                              str(index_left))

                # Beaking the loop
                break

        # Checking right:
        # This is a left max, checking for the right counterpart

        for index_right in range(len(max_val)-index_max+1):
            if right_edge_is_max and (index_max == (len(max_val)-1)):
                # The right edge was manually added as a maximum, no check
                # to the right
                break

            if (index_right == 0) and (index_max == (len(max_val)-1)):
                # This is the most right max
                if most_right_min_pos < max_pos[index_max]:
                    break
                else:
                    right_max_val = potential_well_full[-1]
                    right_max_pos = time_array_full[-1]
            elif (index_right == 1) and (index_max == (len(max_val)-1)):
                # This indexes set is there to avoid checking the right edge
                # twice while one most right max
                continue
            elif (index_right == 0) and (index_max != (len(max_val)-1)):
                # This indexes set corresponds to the same max
                continue

            elif (index_right == (len(max_val)-index_max)) and \
                    (index_max != (len(max_val)-1)):

                # No more max on the right, checking edge
                right_max_val = potential_well_full[-1]
                right_max_pos = time_array_full[-1]
            else:
                right_max_val = max_val[index_max+index_right]
                right_max_pos = max_pos[index_max+index_right]

            left_pos = present_max_pos
            left_val = present_max_val

            if np.isclose(right_max_val, present_max_val,
                          rtol=relative_max_val_precision_limit, atol=0):
                # The right max is identical to the present max, a pot.
                # well is found
                right_pos = right_max_pos
                right_val = right_max_val

                if [left_pos, right_pos] not in potwell_max_locs:
                    potwell_max_locs.append([left_pos, right_pos])
                    potwell_max_vals.append([left_val, right_val])

                    if np.isnan(inner_sep_max_right):
                        potwell_inner_max.append(np.nan)
                        potwell_min_locs.append(
                                float(min_pos[(min_pos > left_pos) *
                                              (min_pos < right_pos)]))
                        potwell_min_vals.append(
                                float(min_val[(min_pos > left_pos) *
                                              (min_pos < right_pos)]))

                    else:
                        potwell_inner_max.append(
                                float(inner_sep_max_right))
                        potwell_min_locs.append(np.nan)
                        potwell_min_vals.append(np.nan)

                    if verbose:
                        print('+R1 - IMAX '+str(index_max)+' - IRIGHT ' +
                              str(index_right))
                        print([left_pos, right_pos], inner_sep_max_right)
                else:
                    pass
                    if verbose:
                        print('=R1 - IMAX '+str(index_max)+' - IRIGHT ' +
                              str(index_right))

                # Breaking the loop
                break

            elif right_max_val < present_max_val:
                # The right max is smaller than the present max, this
                # means that there is an inner separatrix

                inner_sep_max_right = np.nanmax([
                    inner_sep_max_right, right_max_val])

                if verbose:
                    print('R2 - IMAX '+str(index_max)+' - IRIGHT ' +
                          str(index_right))
                    print(inner_sep_max_right)

            elif right_max_val > present_max_val:
                # The right max is higher than the present max, finding
                # the intersection and breaking the loop

                indexes_find_root = np.where(
                    (tck[0] > present_max_pos) *
                    (tck[0] <= (right_max_pos+3*time_resolution)))

                tck_adjusted = (
                    tck[0][indexes_find_root],
                    (tck[1]-present_max_val)[indexes_find_root],
                    tck[2])

                potential_well_roots = interp.sproot(tck_adjusted,
                                                     mest=mest)

                # Breaking if root finding fails (bucket too small or
                # precision param too fine)
                if len(potential_well_roots) == 0:
                    print('Warning: could not intersect potential well ' +
                          'on the right! ' +
                          'Try lowering relative_max_val_precision_limit')
                    break

                right_pos = np.min(potential_well_roots)
                right_val = present_max_val

                if [left_pos, right_pos] not in potwell_max_locs:
                    potwell_max_locs.append([left_pos, right_pos])
                    potwell_max_vals.append([left_val, right_val])

                    if np.isnan(inner_sep_max_right):
                        potwell_inner_max.append(np.nan)
                        potwell_min_locs.append(
                                float(min_pos[(min_pos > left_pos) *
                                              (min_pos < right_pos)]))
                        potwell_min_vals.append(
                                float(min_val[(min_pos > left_pos) *
                                              (min_pos < right_pos)]))

                    else:
                        potwell_inner_max.append(
                                float(inner_sep_max_right))
                        potwell_min_locs.append(np.nan)
                        potwell_min_vals.append(np.nan)

                    if verbose:
                        print('+R3 - IMAX '+str(index_max)+' - IRIGHT ' +
                              str(index_right))
                        print([left_pos, right_pos],
                              inner_sep_max_right)

                else:
                    pass
                    if verbose:
                        print('=R3 - IMAX '+str(index_max)+' - IRIGHT ' +
                              str(index_right))

                # Beaking the loop
                break

    return (potwell_max_locs, potwell_max_vals,
            potwell_inner_max, potwell_min_locs,
            potwell_min_vals)


def potential_well_cut_cubic(time_array_full, potential_well_full,
                             potwell_max_locs):

    tck_potential_well = interp.splrep(time_array_full,
                                       potential_well_full)

    potential_well_list = []
    time_array_list = []
    for index_well in range(len(potwell_max_locs)):

        left_position = potwell_max_locs[index_well][0]
        right_position = potwell_max_locs[index_well][1]

        new_n_points = len(
            time_array_full[
                (time_array_full >= left_position) *
                (time_array_full <= right_position)])

        xnew = np.linspace(left_position, right_position, new_n_points)

        out = interp.splev(xnew, tck_potential_well)

        time_array_list.append(xnew)
        potential_well_list.append(out)

    return time_array_list, potential_well_list


def sort_potential_wells(time_list, well_list, by = 't_start'):

    if not hasattr(time_list[0], '__iter__'):
        time_list = (time_list,)
        well_list = (well_list,)
    
    if by == 't_start':
        order = [a for a,b in sorted(enumerate(time_list), 
                                 key = lambda itt : itt[1][0])]
    elif by == 'size':
        order = [a for a,b in sorted(enumerate(time_list), 
                                 key = lambda itt : itt[1][0] - itt[1][-1])]
    else:
        raise AttributeError("no sorting option for " + str(by))
    
    retTimes = [time_list[i] for i in order]
    retWells = [well_list[i] for i in order]
    
    return retTimes, retWells


def potential_to_hamiltonian(time_array, potential_array, beta, energy, eta):
    
    HVal = np.max(potential_array)
    return np.abs((HVal - potential_array)*2*beta**2*energy/eta)


def trajectory_area_cubic(time_array, potential_array, eta_0, beta_rel,
                          tot_energy, min_potential_well=None):

    if min_potential_well is None:
        min_potential_well = np.min(minmax_location_cubic(
            time_array, potential_array)[1][0])

    eom_factor_dE = abs(eta_0) / (2*beta_rel**2.*tot_energy)

    dEtraj = np.sqrt((potential_array[0]-potential_array) / eom_factor_dE)
    dEtraj[np.isnan(dEtraj)] = 0

    full_length_time = time_array[-1]-time_array[0]
    hamiltonian = potential_array[0]-min_potential_well
    calc_area = 2*integ_cubic(time_array, dEtraj)[1][-1]
    half_energy_height = np.sqrt((hamiltonian) / eom_factor_dE)

    return time_array, dEtraj, hamiltonian, calc_area, half_energy_height, \
        full_length_time


def area_vs_hamiltonian_cubic(time_array, potential_array, eta_0, beta_rel,
                              tot_energy, min_potential_well=None,
                              inner_max_potential_well=None,
                              n_points_reinterp=None):

    if (inner_max_potential_well is not None) and \
            np.isfinite(inner_max_potential_well):
        n_points_above_inner_max = len(potential_array[1:-1][
            potential_array[1:-1] >= inner_max_potential_well])
        index_above_inner_max = np.where(
            potential_array[1:-1] >= inner_max_potential_well)[0]
    else:
        n_points_above_inner_max = len(potential_array)-2
        index_above_inner_max = np.arange(1, len(potential_array)-1)

    tck_potential_well = interp.splrep(time_array, potential_array)

    if min_potential_well is None:
        min_potential_well = np.min(minmax_location_cubic(
            time_array, potential_array, tck=tck_potential_well)[1][0])

    calc_area_scan = np.empty(n_points_above_inner_max)
    calc_area_scan[:] = np.nan
    hamiltonian_scan = np.empty(n_points_above_inner_max)
    hamiltonian_scan[:] = np.nan
    half_energy_height_scan = np.empty(n_points_above_inner_max)
    half_energy_height_scan[:] = np.nan
    full_length_time_scan = np.empty(n_points_above_inner_max)
    full_length_time_scan[:] = np.nan

    for counter, indexAmplitude in enumerate(index_above_inner_max):

        tck_adjusted = (
            tck_potential_well[0],
            tck_potential_well[1]-tck_potential_well[1][indexAmplitude],
            tck_potential_well[2])

        roots_adjusted = interp.sproot(tck_adjusted)

        if len(roots_adjusted) != 2:
            continue

        left_position = np.min(roots_adjusted)
        right_position = np.max(roots_adjusted)

        if n_points_reinterp is None:
            n_points_reinterp = len(np.where(
                (tck_potential_well[0] >= left_position) *
                (tck_potential_well[0] <= right_position))[0])

        fine_time_array = np.linspace(left_position, right_position,
                                      n_points_reinterp)
        fine_potential_well = interp.splev(fine_time_array, tck_adjusted) + \
            tck_potential_well[1][indexAmplitude]

        (time_array_traj, dEtraj,
         hamiltonian, calc_area,
         half_energy_height,
         full_length_time) = trajectory_area_cubic(
             fine_time_array, fine_potential_well, eta_0, beta_rel,
             tot_energy, min_potential_well=min_potential_well)

        calc_area_scan[counter] = calc_area
        hamiltonian_scan[counter] = hamiltonian
        half_energy_height_scan[counter] = half_energy_height
        full_length_time_scan[counter] = full_length_time

    good_indexes = np.isfinite(calc_area_scan)

    return time_array[index_above_inner_max][good_indexes], \
        hamiltonian_scan[good_indexes], calc_area_scan[good_indexes], \
        half_energy_height_scan[good_indexes], \
        full_length_time_scan[good_indexes]


def synchrotron_frequency_cubic(time_array, potential_array, eta_0, beta_rel,
                                tot_energy, min_potential_well=None,
                                inner_max_potential_well=None,
                                n_points_reinterp=None):

    (time_array_ham, hamiltonian_scan,
     calc_area_scan, half_energy_height_scan,
     full_length_time_scan) = area_vs_hamiltonian_cubic(
        time_array, potential_array, eta_0, beta_rel,
        tot_energy, min_potential_well=min_potential_well,
        inner_max_potential_well=inner_max_potential_well,
        n_points_reinterp=n_points_reinterp)

    sync_freq = np.zeros(len(calc_area_scan))
    time_array_fs = np.zeros(len(calc_area_scan))
    hamiltonian_scan_fs = np.zeros(len(calc_area_scan))
    calc_area_scan_fs = np.zeros(len(calc_area_scan))
    half_energy_height_scan_fs = np.zeros(len(calc_area_scan))
    full_length_time_scan_fs = np.zeros(len(calc_area_scan))

    # Taking every second point, in single RF the consecutive points
    # can be too close to each other for cubic spline interpolation
    sorted_area = np.argsort(calc_area_scan)

    sync_freq[::2] = deriv_cubic(
        calc_area_scan[sorted_area][::2],
        hamiltonian_scan[sorted_area][::2])[1]

    time_array_fs[::2] = time_array_ham[sorted_area][::2]
    hamiltonian_scan_fs[::2] = hamiltonian_scan[sorted_area][::2]
    calc_area_scan_fs[::2] = calc_area_scan[sorted_area][::2]
    half_energy_height_scan_fs[::2] = half_energy_height_scan[sorted_area][::2]
    full_length_time_scan_fs[::2] = full_length_time_scan[sorted_area][::2]

    # Doing the same with the second set of points
    
    sync_freq[1::2] = deriv_cubic(
        calc_area_scan[sorted_area][1::2],
        hamiltonian_scan[sorted_area][1::2])[1]

    time_array_fs[1::2] = time_array_ham[sorted_area][1::2]
    hamiltonian_scan_fs[1::2] = hamiltonian_scan[sorted_area][1::2]
    calc_area_scan_fs[1::2] = calc_area_scan[sorted_area][1::2]
    half_energy_height_scan_fs[1::2] = half_energy_height_scan[sorted_area][1::2]
    full_length_time_scan_fs[1::2] = full_length_time_scan[sorted_area][1::2]

    sorted_time = np.argsort(time_array_fs)

    return time_array_fs[sorted_time], \
        sync_freq[sorted_time], \
        hamiltonian_scan_fs[sorted_time], \
        calc_area_scan_fs[sorted_time], \
        half_energy_height_scan_fs[sorted_time], \
        full_length_time_scan_fs[sorted_time]
