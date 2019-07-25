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


def potential_well_generation_cubic(voltage_array, time_array, eta_0,
                                    charge, t_rev, energy_increment,
                                    interpolated_voltage_minus_increment=None):

    eom_factor_potential = np.sign(eta_0) * charge / t_rev

    if interpolated_voltage_minus_increment is None:
        voltage_minus_increment = (
            voltage_array-(-energy_increment)/abs(charge))
        interpolated_voltage_minus_increment = interp.splrep(
            time_array, voltage_minus_increment)
    else:
        pass

    potential_well = (
        -eom_factor_potential*integ_cubic(
            voltage_minus_increment,
            time_array,
            tck=interpolated_voltage_minus_increment))

    return (potential_well, time_array,
            (voltage_minus_increment, interpolated_voltage_minus_increment))


def potential_well_cut_cubic(x, y, der=None, tck=None, tck_der=None, s=0):

    if tck is None:
        tck = interp.splrep(x, y, s=s)
    if tck_der is None and der is None:
        der = deriv_cubic(y, x, tck, s=s)
    if tck_der is None:
        tck_der = interp.splrep(x, der, s=s)

    [min_pos, max_pos], [min_val, max_val] = minmax_location_cubic(
                                               y, x, der=der, tck=tck,
                                               tck_der=tck_der, s=s)

    left_position = 0
    right_position = 0

    lower_maximum = np.where(max_val == np.min([max_val[0], max_val[-1]]))[0]

    precision_parameter = 1e-3

    if len(lower_maximum) > 1:

        left_position = max_pos[0]
        right_position = max_pos[-1]

        adjustment_value = np.mean(max_val)

        tck_adjusted = (tck[0], tck[1]-adjustment_value, tck[2])
        potential_well_roots = interp.sproot(tck_adjusted)

    elif np.abs((max_val[0]-max_val[-1])/max_val[0]) < precision_parameter:

        left_position = max_pos[0]
        right_position = max_pos[-1]

        adjustment_value = np.mean(max_val)

        tck_adjusted = (tck[0], tck[1]-adjustment_value, tck[2])
        potential_well_roots = interp.sproot(tck_adjusted)

    else:
        if lower_maximum == 0:
            adjustment_value = max_val[0]
            left_position = max_pos[0]
        else:
            adjustment_value = max_val[-1]
            right_position = max_pos[-1]

        tck_adjusted = (tck[0], tck[1]-adjustment_value, tck[2])
        potential_well_roots = interp.sproot(tck_adjusted)

        if lower_maximum == 0:
            right_position = potential_well_roots[
                potential_well_roots > min_pos[min_val == np.min(min_val)]][0]
        else:
            left_position = potential_well_roots[
                potential_well_roots < min_pos[min_val == np.min(min_val)]][-1]

    new_n_points = len(x[(x >= left_position)*(x <= right_position)])
    xnew = np.linspace(left_position, right_position, new_n_points)
    out = interp.splev(xnew, (tck[0], tck[1]-np.min(min_val), tck[2]))

    return out, xnew, (tck_adjusted)


def bucket_area(time_array, potential_array, eom_factor_dE=1.):

    if time_array is None:
        time_array = np.arange(len(potential_array), potential_array.dtype())

    dEtraj = np.sqrt((potential_array[0]-potential_array) / eom_factor_dE)
    dEtraj[np.isnan(dEtraj)] = 0

    hamiltonian = potential_array[0]
    calc_area = 2*integ_cubic(dEtraj, time_array)[-1]
    half_energy_height = np.sqrt((hamiltonian) / eom_factor_dE)

    return calc_area, half_energy_height, hamiltonian
