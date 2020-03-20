# General imports
import numpy as np
import matplotlib.pyplot as plt
import warnings
import numbers
import sys

def matched_profile(dist_type, size, time_array, well_array, dE_array, beta, 
                    energy, eta, J_array = None):
    
    if isinstance(dist_type, str):
        if dist_type in ['binomial', 'waterbag', 'parabolic_amplitude',
                 'parabolic_line']:
            if dist_type == 'waterbag':
                exponent = 0
            elif dist_type == 'parabolic_amplitude':
                exponent = 1
            elif dist_type == 'parabolic_line':
                exponent = 0.5
        else:
            raise RuntimeError('invalid distribution type')
    elif isinstance(dist_type, numbers.Number):
        exponent = dist_type
    else:
        raise RuntimeError('invalid distribution type')

    grids = t_H_J_grids(time_array, well_array, dE_array, beta, energy, eta, 
                        J_array)
    
    density_grid = make_density_grid(grids[2], size, well_array, exponent, 
                                     grids[3])
    
    profile = np.sum(density_grid, axis=0)
    energy = np.sum(density_grid, axis=1)

    return profile, energy
    

#Make grid of time, dE, Hamiltonian and Action for given potential well
def t_H_J_grids(time, pot_well, dE_array, beta, energy, eta, J_array = None):

    t_grid, dE_grid = np.meshgrid(time, dE_array)
    H_grid = (np.abs(eta)*dE_grid**2/(2*beta**2*energy)
             + np.repeat(np.array([pot_well]), len(dE_array), axis=0))
    if J_array is not None:
        J_grid = np.interp(H_grid, pot_well[pot_well.argsort()],
                           J_array[pot_well.argsort()], left=0, right=np.inf)
    else:
        J_grid = None

    return t_grid, dE_grid, H_grid, J_grid


#Create a density grid based on given inputs
def make_density_grid(H_grid, size, pot_well, exponent, J_grid = None):

    distribution_function_ = distribution_function

    # Choice of either H or J as the variable used, if J_grid is None
    # H is used for bunch generation
    if J_grid is None:
        X_grid = H_grid
    else:
        X_grid = J_grid
    
    density_grid = distribution_function_(X_grid, size, exponent)

    # Normalizing the grid
    density_grid[H_grid>np.max(pot_well)] = 0
    density_grid = density_grid / np.sum(density_grid)

    return density_grid


def distribution_function(action_array, size, exponent):
    '''
    *Distribution function (formulas from Laclare).*
    '''

    warnings.filterwarnings("ignore")
    distribution_function_ = (1 - action_array / size)**exponent
    warnings.filterwarnings("default")
    distribution_function_[action_array > size] = 0

    return distribution_function_

    # elif dist_type is 'gaussian':
    #     distribution_function_ = np.exp(- 2 * action_array / size)
    #     return distribution_function_

    # else:
    #     raise RuntimeError('The dist_type option was not recognized')