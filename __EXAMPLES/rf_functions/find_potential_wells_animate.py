'''
Animation of the find_potential_wells_cubic function
'''

# Adding folder on TOP of blond_common to PYTHONPATH
import sys
import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt
sys.path.append('./../../../')

from blond_common.maths.calculus import minmax_location_cubic


def find_potential_wells_cubic_animate(time_array_full, potential_well_full,
                                       relative_max_val_precision_limit=1e-5,
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

    deepest_min = np.min(min_val)

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

    plt.figure('Potential well')
    plt.clf()
    plt.title('The potential well')
    plt.plot(time_array_full, potential_well_full)
#     plt.pause(1)
    plt.title('Finding minima')
    for index_min in range(len(min_pos)):
        plt.plot(min_pos[index_min], min_val[index_min], 'go', markersize=3)
#         plt.pause(0.5)
#     plt.pause(1)
    plt.title('Finding maxima')
    for index_max in range(len(max_pos)):
        plt.plot(max_pos[index_max], max_val[index_max], 'ro', markersize=3)
#         plt.pause(0.5)
#     plt.pause(3)

    wells_found = 0
    plt.figure('Potential well')
    plt.clf()
    plt.title('Wells found: %d' % (wells_found))
    plt.plot(time_array_full, potential_well_full)

    for index_max in range(len(max_val)):

        # Setting a max
        present_max_val = max_val[index_max]
        present_max_pos = max_pos[index_max]

        # Resetting the inner separatrix flag to 0
        inner_sep_max_left = np.nan
        inner_sep_max_right = np.nan

        plt.figure('Potential well')
        plt.clf()
        plt.title('Wells found: %d' % (wells_found))
        label = 'Taking a max'
        plt.plot(present_max_pos, present_max_val, 'ro', markersize=3,
                 label=label)
        plt.plot(time_array_full, potential_well_full, zorder=1)
        plt.legend(loc='upper left')
        ax = plt.gca()
        plt.pause(1)

        # Checking left
        # This is a right max, checking for the left counterparts
        for index_left in range(index_max+2):

            if left_edge_is_max and (index_max == 0):
                # The left edge was manually added as a maximum, no check
                # to the left
                break

            if (index_left == 0) and (index_max == 0):
                # This is the most left max
                label += '\nThis is the most left max!'
                ax.legend(labels=(label, ))
                plt.pause(1)
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
                label += '\nChecking the left edge!'
                left_max_val = potential_well_full[0]
                left_max_pos = time_array_full[0]
            else:
                left_max_val = max_val[index_max-index_left]
                left_max_pos = max_pos[index_max-index_left]

            right_pos = present_max_pos
            right_val = present_max_val

            if np.isclose(left_max_val-deepest_min,
                          present_max_val-deepest_min,
                          rtol=relative_max_val_precision_limit, atol=0):
                # The left max is identical to the present max, a pot. well
                # is found
                left_pos = left_max_pos
                left_val = left_max_val

                if [left_pos, right_pos] not in potwell_max_locs:
                    potwell_max_locs.append([left_pos, right_pos])
                    potwell_max_vals.append([left_val, right_val])

                    label += '\nThere is a higher max on the left!'
                    ax.legend(labels=(label, ))
                    plt.pause(1)

                    label += '\nIntersecting!'
                    plt.plot(left_pos, left_val, 'ro',
                             markersize=3, label=label)
                    ax.legend(labels=(label, ))
                    plt.pause(1)

                    if np.isnan(inner_sep_max_left):
                        potwell_inner_max.append(np.nan)
                        potwell_min_locs.append(
                                float(min_pos[(min_pos > left_pos) *
                                              (min_pos < right_pos)]))
                        potwell_min_vals.append(
                                float(min_val[(min_pos > left_pos) *
                                              (min_pos < right_pos)]))

                        label += '\nNo inner sep, taking min!'
                        plt.plot(potwell_min_locs[-1],
                                 potwell_min_vals[-1],
                                 'go', markersize=3, label=label)
                        ax.legend(labels=(label, ))
                        wells_found += 1
                        plt.title('Wells found: %d' % (wells_found))
                        plt.pause(1)

                    else:
                        potwell_inner_max.append(
                                float(inner_sep_max_left))
                        potwell_min_locs.append(np.nan)
                        potwell_min_vals.append(np.nan)

                        label += '\nWith inner sep, taking highest max!'
                        plt.plot([left_pos, right_pos],
                                 [float(inner_sep_max_left),
                                  float(inner_sep_max_left)],
                                 'g', label=label)
                        ax.legend(labels=(label, ))
                        wells_found += 1
                        plt.title('Wells found: %d' % (wells_found))
                        plt.pause(1)

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

                label += '\nThere is a smaller max on the left! ' + \
                    '(inner sep)'
                ax.legend(labels=(label, ))
                plt.pause(1)

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

                    label += '\nThere is a higher max on the left!'
                    ax.legend(labels=(label, ))
                    plt.pause(1)

                    label += '\nIntersecting!'
                    plt.plot(left_pos, left_val, 'ro',
                             markersize=3, label=label)
                    ax.legend(labels=(label, ))
                    plt.pause(1)

                    if np.isnan(inner_sep_max_left):
                        potwell_inner_max.append(np.nan)
                        potwell_min_locs.append(
                                float(min_pos[(min_pos > left_pos) *
                                              (min_pos < right_pos)]))
                        potwell_min_vals.append(
                                float(min_val[(min_pos > left_pos) *
                                              (min_pos < right_pos)]))

                        label += '\nNo inner sep, taking min!'
                        plt.plot(potwell_min_locs[-1],
                                 potwell_min_vals[-1],
                                 'go', markersize=3, label=label)
                        ax.legend(labels=(label, ))
                        wells_found += 1
                        plt.title('Wells found: %d' % (wells_found))
                        plt.pause(1)

                    else:
                        potwell_inner_max.append(
                                float(inner_sep_max_left))
                        potwell_min_locs.append(np.nan)
                        potwell_min_vals.append(np.nan)

                        label += '\nWith inner sep, taking highest max!'
                        plt.plot([left_pos, right_pos],
                                 [float(inner_sep_max_left),
                                  float(inner_sep_max_left)],
                                 'g', label=label)
                        ax.legend(labels=(label, ))
                        wells_found += 1
                        plt.title('Wells found: %d' % (wells_found))
                        plt.pause(1)

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
                label += '\nThis is the most right max!'
                ax.legend(labels=(label, ))
                plt.pause(1)
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
                label += '\nChecking the right edge!'
                right_max_val = potential_well_full[-1]
                right_max_pos = time_array_full[-1]
            else:
                right_max_val = max_val[index_max+index_right]
                right_max_pos = max_pos[index_max+index_right]

            left_pos = present_max_pos
            left_val = present_max_val

            if np.isclose(right_max_val-deepest_min,
                          present_max_val-deepest_min,
                          rtol=relative_max_val_precision_limit, atol=0):
                # The right max is identical to the present max, a pot.
                # well is found
                right_pos = right_max_pos
                right_val = right_max_val

                if [left_pos, right_pos] not in potwell_max_locs:
                    potwell_max_locs.append([left_pos, right_pos])
                    potwell_max_vals.append([left_val, right_val])

                    label += '\nThere is a higher max on the right!'
                    ax.legend(labels=(label, ))
                    plt.pause(1)

                    label += '\nIntersecting!'
                    plt.plot(right_pos, right_val, 'ro',
                             markersize=3, label=label)
                    ax.legend(labels=(label, ))
                    plt.pause(1)

                    if np.isnan(inner_sep_max_right):
                        potwell_inner_max.append(np.nan)
                        potwell_min_locs.append(
                                float(min_pos[(min_pos > left_pos) *
                                              (min_pos < right_pos)]))
                        potwell_min_vals.append(
                                float(min_val[(min_pos > left_pos) *
                                              (min_pos < right_pos)]))

                        label += '\nNo inner sep, taking min!'
                        plt.plot(potwell_min_locs[-1],
                                 potwell_min_vals[-1],
                                 'go', markersize=3, label=label)
                        ax.legend(labels=(label, ))
                        wells_found += 1
                        plt.title('Wells found: %d' % (wells_found))
                        plt.pause(1)

                    else:
                        potwell_inner_max.append(
                                float(inner_sep_max_right))
                        potwell_min_locs.append(np.nan)
                        potwell_min_vals.append(np.nan)

                        label += '\nWith inner sep, taking highest max!'
                        plt.plot([left_pos, right_pos],
                                 [float(inner_sep_max_right),
                                  float(inner_sep_max_right)],
                                 'g', label=label)
                        ax.legend(labels=(label, ))
                        wells_found += 1
                        plt.title('Wells found: %d' % (wells_found))
                        plt.pause(1)

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

                label += '\nThere is a smaller max on the right! ' + \
                    '(inner sep)'
                ax.legend(labels=(label, ))
                plt.pause(1)

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

                    label += '\nThere is a higher max on the right!'
                    ax.legend(labels=(label, ))
                    plt.pause(1)

                    label += '\nIntersecting!'
                    plt.plot(right_pos, right_val, 'ro',
                             markersize=3, label=label)
                    ax.legend(labels=(label, ))
                    plt.pause(1)

                    if np.isnan(inner_sep_max_right):
                        potwell_inner_max.append(np.nan)
                        potwell_min_locs.append(
                                float(min_pos[(min_pos > left_pos) *
                                              (min_pos < right_pos)]))
                        potwell_min_vals.append(
                                float(min_val[(min_pos > left_pos) *
                                              (min_pos < right_pos)]))

                        label += '\nNo inner sep, taking min!'
                        plt.plot(potwell_min_locs[-1],
                                 potwell_min_vals[-1],
                                 'go', markersize=3, label=label)
                        ax.legend(labels=(label, ))
                        wells_found += 1
                        plt.title('Wells found: %d' % (wells_found))
                        plt.pause(1)

                    else:
                        potwell_inner_max.append(
                                float(inner_sep_max_right))
                        potwell_min_locs.append(np.nan)
                        potwell_min_vals.append(np.nan)

                        label += '\nWith inner sep, taking highest max!'
                        plt.plot([left_pos, right_pos],
                                 [float(inner_sep_max_right),
                                  float(inner_sep_max_right)],
                                 'g', label=label)
                        ax.legend(labels=(label, ))
                        wells_found += 1
                        plt.title('Wells found: %d' % (wells_found))
                        plt.pause(1)

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


if __name__ == "__main__":

    # Defining a ramp with a program time vs. energy (warning: initial energy cannot be 0)

    from blond_common.interfaces.beam.beam import Particle
    from blond_common.interfaces.input_parameters.ring import Ring
    from scipy.constants import u, c, e

    ring_length = 2*np.pi*100                   # Machine circumference [m]
    bending_radius = 70.079                     # Bending radius [m]
    bending_field = 1.136487                    # Bending field [T]
    gamma_transition = 6.1                      # Transition gamma
    alpha_0 = 1/gamma_transition**2.

    particle_charge = 39                            # Particle charge [e]
    particle_mass = 128.883*u*c**2./e               # Particle mass [eV/c2]
    particle = Particle(particle_mass, particle_charge)

    ring = Ring(ring_length, alpha_0, bending_field,
                particle, synchronous_data_type='bending field',
                bending_radius=bending_radius)

    from blond_common.interfaces.input_parameters.rf_parameters import RFStation

    harmonic = [21, 28, 169]
    #voltage = [80e3, 0, 0]  # V, h21 Single RF
    voltage = [6e3, 20e3, 0]  # V, h21->h28 batch compression
    voltage = [0, 16.1e3, 12.4e3]  # V, h28->h169 rebucketting
    phi_rf = [np.pi, np.pi, np.pi]  # rad

    rf_station = RFStation(ring, harmonic, voltage, phi_rf, n_rf=3)

    from blond_common.rf_functions.potential import rf_voltage_generation

    n_points = 10000
    t_rev = ring.t_rev[0]
    voltage = rf_station.voltage[:, 0]
    harmonic = rf_station.harmonic[:, 0]
    phi_rf = rf_station.phi_rf[:, 0]
    time_bounds = [-ring.t_rev[0]/harmonic[0]*2, ring.t_rev[0]/harmonic[0]*2]

    time_array, rf_voltage_array = rf_voltage_generation(
        n_points, t_rev, voltage, harmonic, phi_rf, time_bounds=time_bounds)

    from blond_common.rf_functions.potential import rf_potential_generation

    n_points = 10000
    eta_0 = ring.eta_0[0, 0]
    charge = ring.Particle.charge
    energy_increment_bis = charge*5e3

    time_array, rf_potential_array_acc = rf_potential_generation(
        n_points, t_rev, voltage, harmonic, phi_rf, eta_0, charge,
        energy_increment_bis, time_bounds=time_bounds)

    (potential_well_locs, potential_well_vals,
     potential_well_inner_max, potential_well_min,
     potential_well_min_val) = find_potential_wells_cubic_animate(
        time_array, rf_potential_array_acc, mest=200, verbose=False)
