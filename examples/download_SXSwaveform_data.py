import sxs
import numpy as np
from scipy.interpolate import interp1d

# Load a specific version of the catalog for reproducibility
df = sxs.load("dataframe", tag="3.0.0")

# Load a specific simulation
sim = sxs.load("SXS:BBH:0305")

# Obtain data about the horizons
horizons = sim.horizons

# Obtain data about the gravitational-wave strain
h = sim.h

times = h.time
strain = h.data[:,4].real

# interpolate the strain data to a uniform time grid
dt = 0.05
new_times = np.arange(times[0], times[-1], dt)
new_strain = interp1d(times, strain)(new_times)

# Save the waveform data to a text file
np.savetxt("../data/waveform_SXS:BBH:0305_Reh22.txt", 
           np.vstack((new_times, new_strain)))