'''
Interpolation functions and potential well related functions
A. Lasheen
'''

# External imports
import numpy as np
import scipy.interpolate as interp
from ..maths.calculus import integ_cubic, deriv_cubic, minmax_location_cubic


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
                               mest=10, verbose=False):

    potwell_max_locs = []
    potwell_max_vals = []
    potwell_inner_max = []
    potwell_min_locs = []
    potwell_min_vals = []

    tck = interp.splrep(time_array_full, potential_well_full)

    min_max_results = minmax_location_cubic(time_array_full,
                                            potential_well_full,
                                            tck=tck,
                                            mest=mest)

    min_pos = min_max_results[0][0]
    max_pos = min_max_results[0][1]
    min_val = min_max_results[1][0]
    max_val = min_max_results[1][1]

    for index_max in range(len(max_val)):

        # Setting a max
        present_max_val = max_val[index_max]
        present_max_pos = max_pos[index_max]

        # Resetting the inner separatrix flag to 0
        inner_sep_max_left = 0
        inner_sep_max_right = 0

        # Checking left
        if index_max == 0:
            # This is the most left max
            pass
        else:
            # This is a right max, checking for the left counterparts
            for index_left in range(1, index_max+1):
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

                        if inner_sep_max_left > 0:
                            potwell_inner_max.append(
                                    float(inner_sep_max_left))
                            potwell_min_locs.append(np.nan)
                            potwell_min_vals.append(np.nan)
                        else:
                            potwell_inner_max.append(np.nan)
                            deepest_min = np.min(
                                min_pos[(min_pos > left_pos) *
                                        (min_pos < right_pos)])
                            potwell_min_vals.append(
                                float(min_val[
                                    (min_pos > left_pos) *
                                    (min_pos < right_pos)][
                                        min_pos[(min_pos > left_pos) *
                                                (min_pos < right_pos)] ==
                                        deepest_min]))
                            potwell_min_locs.append(
                                    float(deepest_min))

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

                    inner_sep_max_left = np.max([
                        inner_sep_max_left, left_max_val])

                    if verbose:
                        print('L2 - IMAX '+str(index_max)+' - ILEFT ' +
                              str(index_left))
                        print (inner_sep_max_left)

                elif left_max_val > present_max_val:
                    # The left max is higher than the present max, finding
                    # the intersection and breaking the loop

                    indexes_find_root = np.where((tck[0] >= left_max_pos) *
                                                 (tck[0] <= present_max_pos))

                    tck_adjusted = (
                        tck[0][indexes_find_root],
                        (tck[1]-present_max_val)[indexes_find_root],
                        tck[2])

                    potential_well_roots = interp.sproot(tck_adjusted,
                                                         mest=mest)

                    # Breaking if root finding fails (bucket too small or
                    # precision param too fine)
                    if len(potential_well_roots) == 0:
                        print('FL')
                        break

                    left_pos = np.max(potential_well_roots)
                    left_val = present_max_val

                    if [left_pos, right_pos] not in potwell_max_locs:
                        potwell_max_locs.append([left_pos, right_pos])
                        potwell_max_vals.append([left_val, right_val])

                        if inner_sep_max_left > 0:
                            potwell_inner_max.append(
                                    float(inner_sep_max_left))
                            potwell_min_locs.append(np.nan)
                            potwell_min_vals.append(np.nan)
                        else:
                            potwell_inner_max.append(np.nan)
                            deepest_min = np.min(
                                min_pos[(min_pos > left_pos) *
                                        (min_pos < right_pos)])
                            potwell_min_vals.append(
                                float(min_val[
                                    (min_pos > left_pos) *
                                    (min_pos < right_pos)][
                                        min_pos[(min_pos > left_pos) *
                                                (min_pos < right_pos)] ==
                                        deepest_min]))
                            potwell_min_locs.append(
                                    float(deepest_min))

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
        if index_max == len(max_val)-1:
            # This is the most right max
            pass
        else:
            # This is a left max, checking for the right counterpart
            for index_right in range(1, len(max_val)-index_max):
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

                        if inner_sep_max_right > 0:
                            potwell_inner_max.append(
                                    float(inner_sep_max_right))
                            potwell_min_locs.append(np.nan)
                            potwell_min_vals.append(np.nan)
                        else:
                            potwell_inner_max.append(np.nan)
                            deepest_min = np.min(
                                min_pos[(min_pos > left_pos) *
                                        (min_pos < right_pos)])
                            potwell_min_vals.append(
                                float(min_val[
                                    (min_pos > left_pos) *
                                    (min_pos < right_pos)][
                                        min_pos[(min_pos > left_pos) *
                                                (min_pos < right_pos)] ==
                                        deepest_min]))
                            potwell_min_locs.append(
                                    float(deepest_min))

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

                    inner_sep_max_right = np.max([inner_sep_max_right,
                                                  right_max_val])

                    if verbose:
                        print('R2 - IMAX '+str(index_max)+' - IRIGHT ' +
                              str(index_right))
                        print(inner_sep_max_right)

                elif right_max_val > present_max_val:
                    # The right max is higher than the present max, finding
                    # the intersection and breaking the loop

                    indexes_find_root = np.where(
                        (tck[0] >= present_max_pos) *
                        (tck[0] <= right_max_pos))

                    tck_adjusted = (
                        tck[0][indexes_find_root],
                        (tck[1]-present_max_val)[indexes_find_root],
                        tck[2])
                    potential_well_roots = interp.sproot(tck_adjusted,
                                                         mest=mest)

                    # Breaking if root finding fails (bucket too small or
                    # precision param too fine)
                    if len(potential_well_roots) == 0:
                        print('FR')
                        break

                    right_pos = np.min(potential_well_roots)
                    right_val = present_max_val

                    if [left_pos, right_pos] not in potwell_max_locs:
                        potwell_max_locs.append([left_pos, right_pos])
                        potwell_max_vals.append([left_val, right_val])

                        if inner_sep_max_right > 0:
                            potwell_inner_max.append(
                                    float(inner_sep_max_right))
                            potwell_min_locs.append(np.nan)
                            potwell_min_vals.append(np.nan)
                        else:
                            potwell_inner_max.append(np.nan)
                            deepest_min = np.min(
                                min_pos[(min_pos > left_pos) *
                                        (min_pos < right_pos)])
                            potwell_min_vals.append(
                                float(min_val[
                                    (min_pos > left_pos) *
                                    (min_pos < right_pos)][
                                        min_pos[(min_pos > left_pos) *
                                                (min_pos < right_pos)] ==
                                        deepest_min]))
                            potwell_min_locs.append(
                                    float(deepest_min))

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


# Cutting the potential wells according to the previous function output
def potential_well_cut(time_array_full, potential_well_full,
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


def bucket_area(time_array, potential_array, eom_factor_dE=1.):

    if time_array is None:
        time_array = np.arange(len(potential_array), potential_array.dtype())

    dEtraj = np.sqrt((potential_array[0]-potential_array) / eom_factor_dE)
    dEtraj[np.isnan(dEtraj)] = 0

    hamiltonian = potential_array[0]
    calc_area = 2*integ_cubic(dEtraj, time_array)[-1]
    half_energy_height = np.sqrt((hamiltonian) / eom_factor_dE)

    return hamiltonian, calc_area, half_energy_height
