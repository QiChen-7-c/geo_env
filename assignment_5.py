from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

matplotlib.use('Agg')

# Paths
DATA_DIR = Path(
    r'/Users/chenq0d/Library/Mobile Documents/com~apple~CloudDocs/shared_doc/KAUST_course_material/Geo-Environmental Modeling & Analysis/Course_Data/GridSat_Data'
)
OUTPUT_DIR = Path(__file__).resolve().parent

# Jeddah location
jeddah_lat = 21.5
jeddah_lon = 39.2

# AutoEstimator constants
A = 1.1183e11
b = 3.6382e-2
c = 1.2

# Part 2: open one file (06 UTC)
file_06 = DATA_DIR / 'GRIDSAT-B1.2009.11.25.06.v02r01.nc'
dset = xr.open_dataset(file_06, mask_and_scale=False)

# Load irwin_cdr and apply scale/offset
IR = np.array(dset.variables['irwin_cdr']).squeeze().astype('float32')
print('The dimensions of IR are: ',IR.shape)

# Flip to north-up
IR = np.flipud(IR)

# fill = dset['irwin_cdr']._FillValue
# IR[IR == fill] = np.nan

IR = IR * 0.01 + 200.0
# Convert to Celsius
IR = IR - 273.15

# Plot
plt.figure(1)
plt.imshow(IR, extent=[-180.035, 180.035, -70.035, 70.035], aspect='auto', cmap='cividis')
cbar = plt.colorbar()
cbar.set_label('Brightness temperature (degrees Celsius)')
plt.scatter(jeddah_lon, jeddah_lat, color='red', marker='o', label='Jeddah')
plt.legend(loc='lower left')
plt.title('GridSat IR Brightness Temperature (2009-11-25 06:00 UTC)')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'assignment_5_ir_06UTC.png', dpi=300)
plt.close()

# Part 3: rainfall estimation 00–12 UTC
files = sorted(DATA_DIR.glob('GRIDSAT-B1.2009.11.25.*.v02r01.nc'))

lat = dset['lat'].values
lon = dset['lon'].values

# Jeddah box for stats
lat_box = 1.0
lon_box = 1.0
lat_mask = (lat >= jeddah_lat - lat_box) & (lat <= jeddah_lat + lat_box)
lon_mask = (lon >= jeddah_lon - lon_box) & (lon <= jeddah_lon + lon_box)

cumulative_rain = None
records = []

for f in files:
    ds = xr.open_dataset(f, mask_and_scale=False)
    IR = np.array(ds.variables['irwin_cdr']).squeeze().astype('float32')
    fill = ds['irwin_cdr']._FillValue
    IR[IR == fill] = np.nan
    IR = IR * 0.01 + 200.0

    # AutoEstimator (needs Kelvin)
    R = A * np.exp(-b * (IR ** c))

    # Accumulate rainfall (3-hour step)
    if cumulative_rain is None:
        cumulative_rain = np.nan_to_num(R) * 3.0
    else:
        cumulative_rain += np.nan_to_num(R) * 3.0

    # Jeddah stats
    sub_tb = IR[np.ix_(lat_mask, lon_mask)]
    sub_rr = R[np.ix_(lat_mask, lon_mask)]

    t = pd.to_datetime(ds['time'].values[0])
    records.append(
        {
            'time_utc': t,
            'min_tb_k': float(np.nanmin(sub_tb)),
            'mean_tb_k': float(np.nanmean(sub_tb)),
            'max_rr_mmh': float(np.nanmax(sub_rr)),
            'mean_rr_mmh': float(np.nanmean(sub_rr)),
        }
    )

stats = pd.DataFrame(records).sort_values('time_utc')
print(stats)

# Spatial resolution
lat_res_deg = float(np.mean(np.diff(lat)))
lon_res_deg = float(np.mean(np.diff(lon)))
km_per_deg = 111.0
print(f"Lat resolution: {lat_res_deg:.5f} deg (~{lat_res_deg*km_per_deg:.2f} km)")
print(f"Lon resolution: {lon_res_deg:.5f} deg (~{lon_res_deg*km_per_deg:.2f} km at equator)")

# Plot cumulative rainfall as a line chart (Jeddah region mean)
stats['cum_rain_mm'] = (stats['mean_rr_mmh'] * 3.0).cumsum()

plt.figure(figsize=(7, 4))
plt.plot(stats['time_utc'], stats['cum_rain_mm'], marker='o')
plt.grid(True, alpha=0.3)
plt.xlabel('Time (UTC)')
plt.ylabel('Cumulative rainfall (mm)')
plt.title('Cumulative Rainfall near Jeddah (2009-11-25 00–12 UTC)')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'assignment_5_cum_rain_00_12UTC.png', dpi=300)
plt.close()
