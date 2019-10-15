#!/usr/bin/env python
# coding: utf-8

# # Using the calculus module with numerical functions

# In[1]:


# Adding folder on TOP of blond_common to PYTHONPATH
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('./../../../')


# ## 0. Defining base functions

# In[2]:


# Generating sine and cosine functions to play with
# A small n_points allows to see the accuracy of the interpolation scheme

n_points = 20  # 20, 1000
angle_array = np.linspace(0, 3*2*np.pi, n_points)
sin_array = np.sin(angle_array)
cos_array = np.cos(angle_array)


# In[3]:


plt.figure('Base func')
plt.clf()
plt.plot(angle_array, sin_array)
plt.plot(angle_array, cos_array)
plt.plot(angle_array, -cos_array)


# ## 1. Derivatives and integrals
# 
# ### 1.1 Derivatives

# In[4]:


# Performing the derivative using deriv_diff
# NB: the length of the output yprime and x are shorter by 1 array element

from blond_common.maths.calculus import deriv_diff

new_angle_diff, deriv_diff_sin_array = deriv_diff(angle_array, sin_array)

plt.figure('Deriv Diff')
plt.plot(new_angle_diff, deriv_diff_sin_array)
plt.plot(angle_array, cos_array)


# In[5]:


# Performing the derivative using deriv_gradient

from blond_common.maths.calculus import deriv_gradient

angle_array_out, deriv_grad_sin_array = deriv_gradient(angle_array, sin_array)

plt.figure('Deriv Gradient')
plt.plot(angle_array_out, deriv_grad_sin_array)
plt.plot(angle_array, cos_array)


# In[6]:


# Performing the derivative using deriv_cubic

from blond_common.maths.calculus import deriv_cubic

angle_array_out, deriv_cubic_sin_array = deriv_cubic(angle_array, sin_array)

plt.figure('Deriv Cubic')
plt.plot(angle_array_out, deriv_cubic_sin_array)
plt.plot(angle_array, cos_array)


# In[7]:


# Performing the derivative using deriv_cubic
# You can pass directly the nodes for the cubic interpolation
# (e.g. to speed up the function is the nodes are calculated only once)
# NB: passing the tck manually allows to interpolate on a different x array

from blond_common.maths.calculus import deriv_cubic
from scipy.interpolate import splrep

tck = splrep(angle_array, sin_array)

angle_array_out, deriv_cubic_sin_array = deriv_cubic(
    np.linspace(0, 3*2*np.pi, n_points*10), 0, tck=tck)

plt.figure('Deriv Cubic -2')
plt.plot(angle_array_out, deriv_cubic_sin_array)
plt.plot(angle_array, cos_array)


# ### 1.2 Integrals

# In[8]:


# Performing the integral using integ_trapz

from blond_common.maths.calculus import integ_trapz

angle_array_out, integ_trapz_sin_array = integ_trapz(angle_array, sin_array)

plt.figure('Integ trapz')
plt.plot(angle_array_out, integ_trapz_sin_array)
plt.plot(angle_array, -cos_array)


# In[9]:


# Performing the integral using integ_trapz
# NB: an integration constant can be passed

from blond_common.maths.calculus import integ_trapz

angle_array_out, integ_trapz_sin_array = integ_trapz(angle_array, sin_array, constant=-1)

plt.figure('Integ trapz-2')
plt.plot(angle_array_out, integ_trapz_sin_array)
plt.plot(angle_array, -cos_array)


# In[10]:


# Performing the integral using integ_cubic

from blond_common.maths.calculus import integ_cubic

angle_array_out, integ_cubic_sin_array = integ_cubic(angle_array, sin_array)

plt.figure('Integ cubic')
plt.plot(angle_array_out, integ_cubic_sin_array)
plt.plot(angle_array, -cos_array)


# In[11]:


# Performing the integral using integ_cubic
# An integration constant and the nodes of the cubic spline interpolation can be passed
# NB: passing the tck manually allows to interpolate on a different x array

from blond_common.maths.calculus import integ_cubic
from scipy.interpolate import splrep, splantider

tck = splrep(angle_array, sin_array)
tck_ader = splantider(tck)

angle_array_out, integ_cubic_sin_array = integ_cubic(
    np.linspace(0, 3*2*np.pi, n_points*10), 0, tck=tck,
    tck_ader=tck_ader, constant=-1)

plt.figure('Integ cubic-2')
plt.plot(angle_array_out, integ_cubic_sin_array)
plt.plot(angle_array, -cos_array)


# ## 2. Zeros finding

# In[12]:


# Using find_zeros_cubic
# NB: zeros exactly on the edge may not be found, need some margin!

from blond_common.maths.calculus import find_zeros_cubic

roots = find_zeros_cubic(angle_array, sin_array)

plt.figure('Find zeros cubic')
plt.plot(angle_array, sin_array)
plt.plot(roots, np.zeros(len(roots)), 'ro')


# ## 3. Minima and maxima

# In[13]:


# Using minmax_location_discrete

from blond_common.maths.calculus import minmax_location_discrete

minmax_pos, minmax_values = minmax_location_discrete(angle_array, sin_array)

min_pos = minmax_pos[0]
max_pos = minmax_pos[1]
min_val = minmax_values[0]
max_val = minmax_values[1]


# In[14]:


plt.figure('Min max discrete')
plt.plot(angle_array, sin_array)
plt.plot(np.linspace(0, 3*2*np.pi, 1000),
         np.sin(np.linspace(0, 3*2*np.pi, 1000)), 'k--')
plt.plot(min_pos, min_val, 'go')
plt.plot(max_pos, max_val, 'ro')


# In[15]:


# Using minmax_location_cubic

from blond_common.maths.calculus import minmax_location_cubic

minmax_pos, minmax_values = minmax_location_cubic(angle_array, sin_array)

min_pos = minmax_pos[0]
max_pos = minmax_pos[1]
min_val = minmax_values[0]
max_val = minmax_values[1]


# In[16]:


plt.figure('Min max cubic')
plt.plot(angle_array, sin_array)
plt.plot(np.linspace(0, 3*2*np.pi, 1000),
         np.sin(np.linspace(0, 3*2*np.pi, 1000)), 'k--')
plt.plot(min_pos, min_val, 'go')
plt.plot(max_pos, max_val, 'ro')

plt.show()
