# General imports
import numpy as np
import scipy.constants as cont

# BLonD_Common imports
from ..devtools import exceptions as excpt
from ..devtools import assertions as assrt

def _get_mass(rest_mass = None, atomic_mass = None, n_nuc = None):
    
    assrt.single_not_none(rest_mass, atomic_mass, n_nuc, 
                          msg = 'Exactly 1 of rest_mass, atomic_mass and n_nuc '\
                          + 'must be defined', exception = excpt.InputError)
    
    if rest_mass is not None:
        return rest_mass
    
    else:
    
        AMU = cont.physical_constants['atomic mass unit-electron '\
                                      + 'volt relationship'][0]
        if atomic_mass:
            rest_mass = atomic_mass*AMU
        elif n_nuc:
            rest_mass = n_nuc*AMU
            
        return rest_mass


def beta_to_gamma(beta):
    gamma = 1/np.sqrt(1-beta*beta)
    return gamma


def gamma_to_beta(gamma):
    beta = np.sqrt(1-1/gamma**2)
    return beta


def frev_to_beta(frev, circ=None, rad=None):

    if circ:
        beta = circ*frev/cont.c
    elif rad:
        beta = 2*np.pi*rad*frev/cont.c
    else:
        excpt.InputError("Either radius or circumference required " \
                         + "to caluclate beta")

    return beta


def beta_to_frev(beta, circ=None, rad=None):

    assrt.single_none(circ, rad, msg = 'Exactly 1 of circ and rad must be '\
                                      + 'defined', 
                                      exception = excpt.InputError)
    if circ is None:
        circ = 2*np.pi*rad

    frev = (beta*cont.c)/circ

    return frev


def beta_to_trev(beta, circ = None, rad = None):
    
    assrt.single_none(circ, rad, msg = 'Exactly 1 of circ and rad must be '\
                                      + 'defined', 
                                      exception = excpt.InputError)
    if circ is None:
        circ = 2*np.pi*rad
        
    trev = circ/(beta*cont.c)
    
    return trev


def mom_to_beta(mom, rest_mass=None, n_nuc=None, atomic_mass=None):

    rest_mass = _get_mass(rest_mass, n_nuc, atomic_mass)        
    beta = 1/np.sqrt(1+rest_mass**2/mom**2)

    return beta


def mom_to_gamma(mom, rest_mass=None, n_nuc=None, atomic_mass=None):

    rest_mass = _get_mass(rest_mass, n_nuc, atomic_mass)
    gamma = np.sqrt((mom/rest_mass)**2 +1)

    return gamma


def mom_to_frev(mom, rest_mass=None, n_nuc=None, atomic_mass=None, 
                circ=None, rad=None):

    rest_mass = _get_mass(rest_mass, n_nuc, atomic_mass)
    beta = mom_to_beta(mom, rest_mass)
    frev = beta_to_frev(beta, circ, rad)

    return frev


def mom_to_trev(mom, rest_mass=None, n_nuc=None, atomic_mass=None, 
                circ=None, rad=None):

    rest_mass = _get_mass(rest_mass, n_nuc, atomic_mass)
    beta = mom_to_beta(mom, rest_mass)
    trev = beta_to_trev(beta, circ, rad)

    return trev


def momentum_to_energy(mom, rest_mass=None, n_nuc=None, atomic_mass=None):

    rest_mass = _get_mass(rest_mass, n_nuc, atomic_mass)
    energy = np.sqrt(rest_mass**2 + mom**2)

    return energy


def momentum_to_kin_energy(mom, rest_mass=None, n_nuc=None, atomic_mass=None):

    rest_mass = _get_mass(rest_mass, n_nuc, atomic_mass)
    energy = np.sqrt(rest_mass**2 + mom**2)

    return energy - rest_mass


def momentum_to_B_field(mom, bending_radius, charge):
    
    B_field = mom/(bending_radius*charge*cont.c)
    
    return B_field


def energy_to_momentum(energy, rest_mass=None, n_nuc=None, atomic_mass=None):

    rest_mass = _get_mass(rest_mass, n_nuc, atomic_mass)
    mom = np.sqrt(energy**2 - rest_mass**2)

    return mom


def energy_to_kin_energy(energy, rest_mass=None, n_nuc=None, atomic_mass=None):
    
    rest_mass = _get_mass(rest_mass, n_nuc, atomic_mass)

    return energy - rest_mass


def energy_to_B_field(energy, bending_radius, charge, rest_mass=None, 
                      n_nuc=None, atomic_mass=None):

    rest_mass = _get_mass(rest_mass, n_nuc, atomic_mass)
    B_field = np.sqrt(energy**2 - rest_mass**2)/(bending_radius*charge*cont.c)

    return B_field


def energy_to_beta(energy, rest_mass=None, n_nuc=None, atomic_mass=None):
    
    rest_mass = _get_mass(rest_mass, n_nuc, atomic_mass)
    beta = 1/np.sqrt(1+rest_mass**2/(energy**2 - rest_mass**2))

    return beta


def energy_to_gamma(energy, rest_mass=None, n_nuc=None, atomic_mass=None):
    
    rest_mass = _get_mass(rest_mass, n_nuc, atomic_mass)
    gamma = np.sqrt((energy**2 - rest_mass**2)/rest_mass**2 + 1)

    return gamma


def kin_energy_to_momentum(kin_energy, rest_mass=None, n_nuc=None, 
                           atomic_mass=None):

    rest_mass = _get_mass(rest_mass, n_nuc, atomic_mass)
    mom = np.sqrt((rest_mass+kin_energy)**2 - rest_mass**2)

    return mom


def kin_energy_to_energy(kin_energy, rest_mass=None, n_nuc=None, 
                         atomic_mass=None):

    rest_mass = _get_mass(rest_mass, n_nuc, atomic_mass)
    energy = kin_energy + rest_mass
    
    return energy


def kin_energy_to_B_field(kin_energy, bending_radius, charge, rest_mass=None, 
                          n_nuc=None, atomic_mass=None):

    rest_mass = _get_mass(rest_mass, n_nuc, atomic_mass)
    B_field = np.sqrt((rest_mass+kin_energy)**2 - rest_mass**2)\
                /(bending_radius*charge*cont.c)

    return B_field


def B_field_to_momentum(B_Field, bending_radius, charge):
    
    mom = B_Field*bending_radius*charge*cont.c
    return mom


def B_field_to_energy(B_Field, bending_radius, charge, rest_mass=None, 
                          n_nuc=None, atomic_mass=None):

    rest_mass = _get_mass(rest_mass, n_nuc, atomic_mass)
    energy = np.sqrt((B_Field*bending_radius*charge*cont.c)**2 \
                     + rest_mass**2)
    
    return energy


def B_field_to_kin_energy(B_Field, bending_radius, charge, rest_mass=None, 
                          n_nuc=None, atomic_mass=None):

    rest_mass = _get_mass(rest_mass, n_nuc, atomic_mass)
    kin_energy = np.sqrt((B_Field*bending_radius*charge*cont.c)**2 \
                         + rest_mass**2) - rest_mass
    
    return kin_energy


def delta_P_to_delta_E(deltaP, momentum, rest_mass=None, n_nuc=None, atomic_mass=None):

    rest_mass = _get_mass(rest_mass, n_nuc, atomic_mass)

    energy = momentum_to_energy(momentum, rest_mass)
    deltaE = energy - np.sqrt(rest_mass**2 + (deltaP + momentum)**2)
    
    return deltaE

def delta_E_to_delta_P(deltaE, energy, rest_mass=None, n_nuc=None, atomic_mass=None):

    rest_mass = _get_mass(rest_mass, n_nuc, atomic_mass)

    momentum = energy_to_momentum(energy, rest_mass)
    deltaP = energy - np.sqrt(rest_mass**2 + (deltaE + momentum)**2)
    
    return deltaP