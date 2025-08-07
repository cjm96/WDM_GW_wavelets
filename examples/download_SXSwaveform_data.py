import sxs
import numpy as np

# Load a specific version of the catalog for reproducibility
df = sxs.load("dataframe", tag="3.0.0")

# Load a specific simulation
sim = sxs.load("SXS:BBH:0305")

# Obtain data about the horizons
horizons = sim.horizons

# Obtain data about the gravitational-wave strain
h = sim.h

# Save the waveform data to a text file
np.savetxt("../data/waveform_SXS:BBH:0305_Reh22.txt", 
           np.vstack((h.time, h.data[:,4].real)))