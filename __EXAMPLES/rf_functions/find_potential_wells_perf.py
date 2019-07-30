'''
Script to evaluate performance of find_potential_well functions
'''

# Adding folder on TOP of blond_common to PYTHONPATH
import sys
import numpy as np
import time
sys.path.append('./../../../')

from blond_common.rf_functions.potential import find_potential_wells_cubic


if __name__ == "__main__":

    # Defining a ramp with a program time vs. energy (warning: initial energy cannot be 0)

    from blond_common_fork.interfaces.beam.beam import Particle
    from blond_common_fork.interfaces.input_parameters.ring import Ring
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

    from blond_common_fork.interfaces.input_parameters.rf_parameters import RFStation

    harmonic = [21, 28, 169]
    #voltage = [80e3, 0, 0]  # V, h21 Single RF
    voltage = [6e3, 20e3, 0]  # V, h21->h28 batch compression
    voltage = [0, 16.1e3, 12.4e3]  # V, h28->h169 rebucketting
    phi_rf = [np.pi, np.pi, np.pi]  # rad

    rf_station = RFStation(ring, harmonic, voltage, phi_rf, n_rf=3)

    from blond_common_fork.rf_functions.potential import rf_voltage_generation

    n_points = 10000
    t_rev = ring.t_rev[0]
    voltage = rf_station.voltage[:, 0]
    harmonic = rf_station.harmonic[:, 0]
    phi_rf = rf_station.phi_rf[:, 0]
    time_bounds = [-ring.t_rev[0]/harmonic[0]*2, ring.t_rev[0]/harmonic[0]*2]

    time_array, rf_voltage_array = rf_voltage_generation(
        n_points, t_rev, voltage, harmonic, phi_rf, time_bounds=time_bounds)

    from blond_common_fork.rf_functions.potential import rf_potential_generation

    n_points = 10000
    eta_0 = ring.eta_0[0, 0]
    charge = ring.Particle.charge
    energy_increment_bis = charge*5e3

    time_array, rf_potential_array_acc = rf_potential_generation(
        n_points, t_rev, voltage, harmonic, phi_rf, eta_0, charge,
        energy_increment_bis, time_bounds=time_bounds)

    t0 = time.perf_counter()

    for iteration in range(1000):

        (potential_well_locs, potential_well_vals,
         potential_well_inner_max, potential_well_min,
         potential_well_min_val) = find_potential_wells_cubic(
            time_array, rf_potential_array_acc, mest=200, verbose=False)

    t1 = time.perf_counter()

    print('Time per call: %.3e ms' % ((t1-t0)/1000*1e3))
    print('Number of potential wells found: %d' % (len(potential_well_locs)))
