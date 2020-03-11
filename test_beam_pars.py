import numpy as np
import matplotlib.pyplot as plt
import sys

import blond_common.interfaces.beam.beam_parameters as beamPars
import blond_common.interfaces.input_parameters.ring as ring
import blond_common.interfaces.input_parameters.rf_parameters as rfPars
import blond_common.rf_functions.potential as pot


sys.exit()

#%%

momentum = [[0.275, 0.805], [1E9, 2E9]]
gammaT = 4.1
alpha = 1/gammaT**2

psb = ring.Ring(2*np.pi*25, alpha, momentum, 'proton', interp_time = 5E-3)

rf = rfPars.RFStation(psb, [1, 2], [8E3, 8E3], [np.pi, np.pi])

#%%

beam = beamPars.Beam_Parameters(psb, rf)

#%%

plt.plot(beam.ring.cycle_time*1E3, beam.areas[0])
plt.xlabel("Cycle time (ms)")
plt.ylabel("Bucket area (eVs)")
plt.show()

plt.plot(beam.ring.cycle_time*1E3, beam.lengths[0]*1E9)
plt.xlabel("Cycle time (ms)")
plt.ylabel("Bucket length (ns)")
plt.show()

plt.plot(beam.ring.cycle_time*1E3, beam.heights[0]/1E6)
plt.xlabel("Cycle time (ms)")
plt.ylabel("Bucket half height (MeV)")
plt.show()

plt.plot(beam.ring.cycle_time*1E3, beam.particle_tracks[0]*1E9)
plt.xlabel("Cycle time (ms)")
plt.ylabel("Synchronous phase (ns)")
plt.show()

#%%

for i in beam.ring.use_turns:
    bucket = beam.buckets[i, 0]
    plt.plot(bucket.separatrix[0]*1E9, bucket.separatrix[1]/1E6)
plt.show()

#%%

beam = beamPars.Beam_Parameters(psb, rf, bunch_emittance = 1)

#%%

plt.plot(beam.ring.cycle_time*1E3, beam.bunch_emittances[0])
plt.xlabel("Cycle time (ms)")
plt.ylabel("Matched area (eVs)")
plt.show()

plt.plot(beam.ring.cycle_time*1E3, beam.bunch_lengths[0]*1E9)
plt.xlabel("Cycle time (ms)")
plt.ylabel("Bunch length (ns)")
plt.show()

plt.plot(beam.ring.cycle_time*1E3, beam.bunch_heights[0]/1E6)
plt.xlabel("Cycle time (ms)")
plt.ylabel("Bunch half height (MeV)")
plt.show()

#%%

beam = beamPars.Beam_Parameters(psb, rf, bunch_emittance = [[0.275, 0.805], [1, 2]])

#%%

plt.plot(beam.ring.cycle_time*1E3, beam.bunch_emittances[0])
plt.xlabel("Cycle time (ms)")
plt.ylabel("Matched area (eVs)")
plt.show()

plt.plot(beam.ring.cycle_time*1E3, beam.bunch_lengths[0]*1E9)
plt.xlabel("Cycle time (ms)")
plt.ylabel("Bunch length (ns)")
plt.show()

plt.plot(beam.ring.cycle_time*1E3, beam.bunch_heights[0]/1E6)
plt.xlabel("Cycle time (ms)")
plt.ylabel("Bunch half height (MeV)")
plt.show()


#%%

rf = rfPars.RFStation(psb, [4, 10], [10E3, 12E3], [np.pi, np.pi])

#%%

beam = beamPars.Beam_Parameters(psb, rf, 
                            bunch_emittance = ([[0.275, 0.805], [0.3, 0.4]], 
                                               [[0.275, 0.805], [0.2, 0.3]]), 
                                init_coord = [100E-9, 650E-9])

#%%

plt.plot(beam.areas[0])
plt.plot(beam.areas[1])
plt.show()

plt.plot(beam.particle_tracks[0])
plt.plot(beam.particle_tracks[1])
plt.show()


plt.plot(beam.ring.cycle_time*1E3, beam.bunch_emittances[0])
plt.plot(beam.ring.cycle_time*1E3, beam.bunch_emittances[1])
plt.xlabel("Cycle time (ms)")
plt.ylabel("Matched area (eVs)")
plt.show()

plt.plot(beam.ring.cycle_time*1E3, beam.bunch_lengths[0]*1E9)
plt.plot(beam.ring.cycle_time*1E3, beam.bunch_lengths[1]*1E9)
plt.xlabel("Cycle time (ms)")
plt.ylabel("Bunch length (ns)")
plt.show()

plt.plot(beam.ring.cycle_time*1E3, beam.bunch_heights[0]/1E6)
plt.plot(beam.ring.cycle_time*1E3, beam.bunch_heights[1]/1E6)
plt.xlabel("Cycle time (ms)")
plt.ylabel("Bunch half height (MeV)")
plt.show()