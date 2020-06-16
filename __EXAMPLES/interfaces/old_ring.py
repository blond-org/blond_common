#!/usr/bin/env python
# coding: utf-8

# # Using interfaces.input_parameters to setup and organize the synchrotron and rf parameters

# In[1]:


# Adding folder on TOP of blond_common to PYTHONPATH
import sys
import numpy as np
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
sys.path.append('./../../../')


# ## 1. Defining the Ring object, where the base synchrotron parameters are defined

# ## 1.1 Single parameters

# In[2]:


# Simplest case: define a synchrotron with constant momentum

from blond_common.interfaces.input_parameters.ring import Ring
from blond_common.interfaces.beam.beam import Proton

ring_length = 6911.5  # m
gamma_transition = 17.95
alpha_0 = 1/gamma_transition**2.
momentum = 25.92e9  # eV/c
particle = Proton()
ring = Ring(ring_length, alpha_0, momentum, particle)

# Note that size of programs is (n_sections, n_turns+1)
print('Momentum : %.5e eV/c' %(ring.momentum[0,0]))
print('Kinetic energy : %.5e eV' %(ring.kin_energy[0,0]))
print('Total energy : %.5e eV' %(ring.energy[0,0]))
print('beta : %.5f' %(ring.beta[0,0]))
print('gamma : %.5f' %(ring.gamma[0,0]))
print('Revolution period : %.5e s' %(ring.t_rev[0]))


# In[3]:


# Now we define a synchrotron by giving the kinetic energy

from blond_common.interfaces.input_parameters.ring import Ring
from blond_common.interfaces.beam.beam import Proton

ring_length = 6911.5  # m
gamma_transition = 17.95
alpha_0 = 1/gamma_transition**2.
kinetic_energy = 25.e9  # eV/c
particle = Proton()
ring = Ring(ring_length, alpha_0, kinetic_energy, particle,
            synchronous_data_type='kinetic energy')

# Note that size of programs is (n_sections, n_turns+1)
print('Momentum : %.5e eV/c' %(ring.momentum[0,0]))
print('Kinetic energy : %.5e eV' %(ring.kin_energy[0,0]))
print('Total energy : %.5e eV' %(ring.energy[0,0]))
print('beta : %.5f' %(ring.beta[0,0]))
print('gamma : %.5f' %(ring.gamma[0,0]))
print('Revolution period : %.5e s' %(ring.t_rev[0]))


# In[4]:


# Now we define a synchrotron by giving the total energy

from blond_common.interfaces.input_parameters.ring import Ring
from blond_common.interfaces.beam.beam import Proton

ring_length = 6911.5  # m
gamma_transition = 17.95
alpha_0 = 1/gamma_transition**2.
total_energy = 25.94e9  # eV
particle = Proton()
ring = Ring(ring_length, alpha_0, total_energy, particle,
            synchronous_data_type='total energy')

# Note that size of programs is (n_sections, n_turns+1)
print('Momentum : %.5e eV/c' %(ring.momentum[0,0]))
print('Kinetic energy : %.5e eV' %(ring.kin_energy[0,0]))
print('Total energy : %.5e eV' %(ring.energy[0,0]))
print('beta : %.5f' %(ring.beta[0,0]))
print('gamma : %.5f' %(ring.gamma[0,0]))
print('Revolution period : %.5e s' %(ring.t_rev[0]))


# In[5]:


# Now we define a synchrotron by giving the magnetic field

from blond_common.interfaces.input_parameters.ring import Ring
from blond_common.interfaces.beam.beam import Proton

ring_length = 6911.5  # m
gamma_transition = 17.95
alpha_0 = 1/gamma_transition**2.
bending_radius = 741  # m
b_field = 0.1166  # T
particle = Proton()
ring = Ring(ring_length, alpha_0, b_field, particle,
            bending_radius=bending_radius,
            synchronous_data_type='bending field')

# Note that size of programs is (n_sections, n_turns+1)
print('Momentum : %.5e eV/c' %(ring.momentum[0,0]))
print('Kinetic energy : %.5e eV' %(ring.kin_energy[0,0]))
print('Total energy : %.5e eV' %(ring.energy[0,0]))
print('beta : %.5f' %(ring.beta[0,0]))
print('gamma : %.5f' %(ring.gamma[0,0]))
print('Revolution period : %.5e s' %(ring.t_rev[0]))


# In[6]:


# Defining an energy increment over one turn

from blond_common.interfaces.input_parameters.ring import Ring
from blond_common.interfaces.beam.beam import Proton

ring_length = 6911.5  # m
gamma_transition = 17.95
alpha_0 = 1/gamma_transition**2.
total_energy = 25.94e9  # eV
delta_E = 1e6  #eV
particle = Proton()
ring = Ring(ring_length, alpha_0, [total_energy, total_energy+delta_E],
            particle,
            synchronous_data_type='total energy')

# Note that size of programs is (n_sections, n_turns+1)
print('Momentum : %.5e eV/c' %(ring.momentum[0,0]))
print('Kinetic energy : %.5e eV' %(ring.kin_energy[0,0]))
print('Total energy : %.5e eV' %(ring.energy[0,0]))
print('Energy gain : %.5e eV' %(ring.delta_E[0,0]))
print('beta : %.5f' %(ring.beta[0,0]))
print('gamma : %.5f' %(ring.gamma[0,0]))
print('Revolution period : %.5e s' %(ring.t_rev[0]))


# In[7]:


# Non-linear momencum compaction factor can also be defined (slippage internally calculated)

from blond_common.interfaces.input_parameters.ring import Ring
from blond_common.interfaces.beam.beam import Proton

ring_length = 6911.5  # m
gamma_transition = 17.95
alpha_0 = 1/gamma_transition**2.
alpha_1 = 1e-4
alpha_2 = 1e-5
momentum = 25.92e9  # eV/c
particle = Proton()
ring = Ring(ring_length, alpha_0, momentum, particle,
           alpha_1=alpha_1, alpha_2=alpha_2)

# Note that size of the array is (n_sections, n_turns+1)
print('Momentum : %.5e eV/c' %(ring.momentum[0,0]))
print('alpha_0 : %.5e' %(ring.alpha_0[0,0]))
print('alpha_1 : %.5e' %(ring.alpha_1[0,0]))
print('alpha_2 : %.5e' %(ring.alpha_2[0,0]))
print('eta_0 : %.5e' %(ring.eta_0[0,0]))
print('eta_1 : %.5e' %(ring.eta_1[0,0]))
print('eta_2 : %.5e' %(ring.eta_2[0,0]))


# ## 1.2 Defining momentum programs

# In[8]:


# Defining a ramp with user defined momentum turn-by-turn program

from blond_common.interfaces.input_parameters.ring import Ring
from blond_common.interfaces.beam.beam import Proton

ring_length = 6911.5  # m
gamma_transition = 17.95
alpha_0 = 1/gamma_transition**2.

energy_init = 26e9  # eV
energy_fin = 27e9  # eV
delta_E = 1e6  # eV
total_energy = np.arange(energy_init, energy_fin, delta_E)  # eV
n_turns = len(total_energy)-1

particle = Proton()
ring = Ring(ring_length, alpha_0, total_energy,
            particle, n_turns=n_turns,
            synchronous_data_type='total energy')


# In[9]:


# Note that size of programs is (n_sections, n_turns+1)
plt.figure('Programs-1')
plt.clf()
plt.plot(ring.cycle_time, ring.energy[0,:], label='Tot. energy')
plt.plot(ring.cycle_time, ring.momentum[0,:], label='Momentum$\\times$c')
plt.plot(ring.cycle_time, ring.kin_energy[0,:], label='Kin. energy')
plt.xlabel('Time [s]')
plt.ylabel('Energy [eV]')
plt.legend(loc='best')


# In[10]:


plt.figure('Programs-1-bis')
plt.clf()
plt.plot(ring.cycle_time, ring.t_rev[:])
plt.xlabel('Time [s]')
plt.ylabel('Revolution period [s]')
plt.twinx()
plt.plot(ring.cycle_time, ring.gamma[0,:], 'r')
plt.ylabel('$\\gamma$')


# In[11]:


# Defining a ramp with a program time vs. energy (warning: initial energy cannot be 0)

from blond_common.interfaces.input_parameters.ring import Ring
from blond_common.interfaces.beam.beam import Proton

ring_length = 6911.5  # m
gamma_transition = 17.95
alpha_0 = 1/gamma_transition**2.

time_start_ramp = 1e-3  # s 
time_end_ramp = 2e-3  # s
time_array = np.array([0, time_start_ramp, time_end_ramp, time_end_ramp+1e-3])  # s

energy_init = 26e9  # eV
energy_fin = 27e9  # eV
delta_E = 1e6  # eV
total_energy = np.array([energy_init, energy_init, energy_fin, energy_fin])  # eV

particle = Proton()
ring = Ring(ring_length, alpha_0, (time_array, total_energy),
            particle,
            synchronous_data_type='total energy')


# In[12]:


# Note that size of programs is (n_sections, n_turns+1)
plt.figure('Programs-2')
plt.clf()
plt.plot(ring.cycle_time, ring.energy[0,:], label='Tot. energy')
plt.plot(ring.cycle_time, ring.momentum[0,:], label='Momentum$\\times$c')
plt.plot(ring.cycle_time, ring.kin_energy[0,:], label='Kin. energy')
plt.xlabel('Time [s]')
plt.ylabel('Energy [eV]')
plt.legend(loc='best')


# In[13]:


plt.figure('Programs-2-bis')
plt.clf()
plt.plot(ring.cycle_time, ring.t_rev[:])
plt.xlabel('Time [s]')
plt.ylabel('Revolution period [s]')
plt.twinx()
plt.plot(ring.cycle_time, ring.gamma[0,:], 'r')
plt.ylabel('$\\gamma$')


# In[14]:


# Defining a momentum compaction program (same can be done for alpha_1 and alpha_2)

from blond_common.interfaces.input_parameters.ring import Ring
from blond_common.interfaces.beam.beam import Proton

ring_length = 6911.5  # m

gamma_transition = np.arange(17.95, 19., 0.01)
alpha_0 = 1/gamma_transition**2.
n_turns = len(alpha_0)-1

total_energy = 26e9  # eV
particle = Proton()
ring = Ring(ring_length, alpha_0, total_energy,
            particle, n_turns=n_turns,
            synchronous_data_type='total energy')


# In[15]:


# Note that size of programs is (n_sections, n_turns+1)
plt.figure('Programs-3')
plt.clf()
plt.plot(ring.cycle_time, ring.alpha_0[0,:])  
plt.xlabel('Time [s]')
plt.ylabel('Linear momentum compaction')


# ## 1.3 Defining a Ring in several sections (e.g. for several rf stations, divide synchrotron radiation in several steps...)

# In[16]:


# Defining a ramp with user defined momentum turn-by-turn program
# Note that the functions to pass a time vs. momentum program for several sections is not implemented yet

from blond_common.interfaces.input_parameters.ring import Ring
from blond_common.interfaces.beam.beam import Proton

ring_length = 6911.5  # m
length_section_1 = ring_length/2  # m
length_section_2 = ring_length/2  # m

gamma_transition = 17.95
alpha_0 = 1/gamma_transition**2.

energy_init = 26e9  # eV
energy_fin = 26.01e9  # eV
delta_E = 1e6  # eV
total_energy_1 = np.arange(energy_init, energy_fin, delta_E)  # eV
total_energy_2 = np.arange(energy_init, energy_fin, delta_E) + delta_E/2 # eV

n_turns = len(total_energy_1)-1
particle = Proton()
ring = Ring([length_section_1, length_section_2], alpha_0,
            [total_energy_1, total_energy_2],
            particle, n_turns=n_turns, n_sections=2,
            synchronous_data_type='total energy')


# In[17]:


# Note that size of programs is (n_sections, n_turns+1)
plt.figure('Programs-4')
plt.clf()
plt.plot(ring.cycle_time, ring.energy[0,:], '.', label='Tot. energy sect. 1')
plt.plot(ring.cycle_time+ring.t_rev[:]/2, ring.energy[1,:], '.', label='Tot. energy sect. 2')
plt.xlabel('Time [s]')
plt.ylabel('Energy [eV]')
plt.legend(loc='best')


# ## 1.4 Using RingOptions for custom ramp definition

# In[18]:


# Defining a ramp with a program time vs. energy (warning: initial energy cannot be 0)

from blond_common.interfaces.beam.beam import Proton

ring_length = 6911.5  # m
gamma_transition = 17.95
alpha_0 = 1/gamma_transition**2.

time_start_ramp = 1e-3  # s 
time_end_ramp = 29e-3  # s
time_array = np.array([0, time_start_ramp, time_end_ramp, time_end_ramp+1e-3])  # s

energy_init = 26e9  # eV
energy_fin = 27e9  # eV
delta_E = 1e6  # eV
total_energy = np.array([energy_init, energy_init, energy_fin, energy_fin])  # eV

particle = Proton()


# In[19]:


# Now adjusting the RingOptions, here with default parameters

from blond_common.interfaces.input_parameters.ring import Ring, RingOptions

ring_options = RingOptions(
    interpolation='linear', flat_bottom=0,
    flat_top=0, t_start=None, t_end=None)

ring = Ring(ring_length, alpha_0, (time_array, total_energy),
            particle, synchronous_data_type='total energy',
            RingOptions=ring_options)


# In[20]:


# Note that size of programs is (n_sections, n_turns+1)
plt.figure('Programs-5')
plt.clf()
plt.plot(time_array, total_energy, 'ro', label='Input')
plt.plot(ring.cycle_time, ring.energy[0,:], '.', label='Output', markersize=0.5)
plt.xlabel('Time [s]')
plt.ylabel('Energy [eV]')
plt.legend(loc='best')


# In[21]:


# Now adjusting the RingOptions, the program is interpolated from the pdot
# Note that the derivative method is np.gradient

from blond_common.interfaces.input_parameters.ring import Ring, RingOptions

ring_options = RingOptions(interpolation='derivative')

ring = Ring(ring_length, alpha_0, (time_array, total_energy),
            particle, synchronous_data_type='total energy',
            RingOptions=ring_options)


# In[22]:


# Note that size of programs is (n_sections, n_turns+1)
plt.figure('Programs-6')
plt.clf()
plt.plot(time_array, total_energy, 'ro', label='Input')
plt.plot(ring.cycle_time, ring.energy[0,:], '.', label='Output', markersize=0.5)
plt.xlabel('Time [s]')
plt.ylabel('Energy [eV]')
plt.legend(loc='best')


# In[23]:


# Now adjusting the RingOptions, adding 1000 turns on the flat bottom and on the flat top

from blond_common.interfaces.input_parameters.ring import Ring, RingOptions

ring_options = RingOptions(flat_bottom=1000,
                           flat_top=1000)

ring = Ring(ring_length, alpha_0, (time_array, total_energy),
            particle, synchronous_data_type='total energy',
            RingOptions=ring_options)


# In[24]:


# Note that size of programs is (n_sections, n_turns+1)
plt.figure('Programs-7')
plt.clf()
plt.plot(time_array, total_energy, 'ro', label='Input')
plt.plot(ring.cycle_time-ring.t_rev[0]*1000, ring.energy[0,:], '.', label='Output', markersize=0.5)
plt.xlabel('Time [s]')
plt.ylabel('Energy [eV]')
plt.legend(loc='best')


# In[25]:


# Now adjusting the RingOptions, simulating only part of the program

from blond_common.interfaces.input_parameters.ring import Ring, RingOptions

ring_options = RingOptions(t_start=0.5e-3,
                           t_end=6e-3)

ring = Ring(ring_length, alpha_0, (time_array, total_energy),
            particle, synchronous_data_type='total energy',
            RingOptions=ring_options)


# In[26]:


# Note that size of programs is (n_sections, n_turns+1)
plt.figure('Programs-8')
plt.clf()
plt.plot(time_array, total_energy, 'ro', label='Input')
plt.plot(ring.cycle_time+0.5e-3, ring.energy[0,:], '.', label='Output', markersize=0.5)
plt.xlabel('Time [s]')
plt.ylabel('Energy [eV]')
plt.legend(loc='best')


# In[27]:


# Now adjusting the RingOptions, calculating the program every ms instead of every turn

from blond_common.interfaces.input_parameters.ring import Ring, RingOptions

ring_options = RingOptions(interp_time=1e-3)

ring = Ring(ring_length, alpha_0, (time_array, total_energy),
            particle, synchronous_data_type='total energy',
            RingOptions=ring_options)


# In[28]:


# Note that size of programs is (n_sections, n_turns+1)
plt.figure('Programs-9')
plt.clf()
plt.plot(time_array, total_energy, 'ro', label='Input')
plt.plot(ring.cycle_time, ring.energy[0,:], '.', label='Output')
plt.xlabel('Time [s]')
plt.ylabel('Energy [eV]')
plt.legend(loc='best')


# In[29]:


# Now adjusting the RingOptions, calculating the program every ms instead of every turn

from blond_common.interfaces.input_parameters.ring import Ring, RingOptions

time_custom = np.linspace(0.5e-3, 7.4e-3, 20)
ring_options = RingOptions(interp_time=time_custom)

ring = Ring(ring_length, alpha_0, (time_array, total_energy),
            particle, synchronous_data_type='total energy',
            RingOptions=ring_options)


# In[30]:


# Note that size of programs is (n_sections, n_turns+1)
plt.figure('Programs-10')
plt.clf()
plt.plot(time_array, total_energy, 'ro', label='Input')
plt.plot(ring.cycle_time+time_custom[0], ring.energy[0,:], '.', label='Output')
plt.xlabel('Time [s]')
plt.ylabel('Energy [eV]')
plt.legend(loc='best')

