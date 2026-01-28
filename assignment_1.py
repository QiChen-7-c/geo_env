import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import xarray as xr

# Loading the SRTM DEM data
dset = xr.open_dataset(r'/Users/chenq0d/Library/Mobile Documents/com~apple~CloudDocs/shared_doc/KAUST_course_material/Geo-Environmental Modeling & Analysis/Course_Data/SRTMGL1_NC.003_Data/N21E039.SRTMGL1_NC.nc')

pdb.set_trace() #checkpoint 1

# Extracting the DEM variable
DEM = np.array(dset.variables['SRTMGL1_DEM'])
dset.close()

pdb.set_trace() #checkpoint 2

# Plotting the DEM and saving the figure
plt.imshow(DEM)
cbar = plt.colorbar()
cbar.set_label('Elevation (m asl)')
plt.show()
plt.savefig('assignment 1.png', dpi=300)