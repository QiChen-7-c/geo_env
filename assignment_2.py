import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import xarray as xr

# Part 1: Importing Climate Model Output
dset_1950_2014 = xr.open_dataset(r'/Users/chenq0d/Library/Mobile Documents/com~apple~CloudDocs/shared_doc/KAUST_course_material/Geo-Environmental Modeling & Analysis/Course_Data/Climate_Model_Data/tas_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_195001-201412.nc')
# pdb.set_trace()

# Part3
dset_1850_1949 = xr.open_dataset(r'/Users/chenq0d/Library/Mobile Documents/com~apple~CloudDocs/shared_doc/KAUST_course_material/Geo-Environmental Modeling & Analysis/Course_Data/Climate_Model_Data/tas_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_185001-194912.nc')

dset_hist = xr.concat(
    [dset_1850_1949, dset_1950_2014],
    dim='time',
    data_vars='minimal',
    coords='minimal',
    compat='override',
    join='override',
)
mean_1850_1900 = dset_hist['tas'].sel(time=slice('1850-01-01', '1900-12-31')).mean('time')
mean_1850_1900_np = np.array(mean_1850_1900)
print('The properties of the variable:')
print(f"Shape: {mean_1850_1900_np.shape}")
print(f"Data type: {mean_1850_1900_np.dtype}")

# Part 3: Scenario means (2071-2100) and difference maps
dset_ssp119 = xr.open_dataset(r'/Users/chenq0d/Library/Mobile Documents/com~apple~CloudDocs/shared_doc/KAUST_course_material/Geo-Environmental Modeling & Analysis/Course_Data/Climate_Model_Data/tas_Amon_GFDL-ESM4_ssp119_r1i1p1f1_gr1_201501-210012.nc')
dset_ssp245 = xr.open_dataset(r'/Users/chenq0d/Library/Mobile Documents/com~apple~CloudDocs/shared_doc/KAUST_course_material/Geo-Environmental Modeling & Analysis/Course_Data/Climate_Model_Data/tas_Amon_GFDL-ESM4_ssp245_r1i1p1f1_gr1_201501-210012.nc')
dset_ssp585 = xr.open_dataset(r'/Users/chenq0d/Library/Mobile Documents/com~apple~CloudDocs/shared_doc/KAUST_course_material/Geo-Environmental Modeling & Analysis/Course_Data/Climate_Model_Data/tas_Amon_GFDL-ESM4_ssp585_r1i1p1f1_gr1_201501-210012.nc')

mean_2071_2100_ssp119 = dset_ssp119['tas'].sel(time=slice('2071-01-01', '2100-12-31')).mean('time')
mean_2071_2100_ssp245 = dset_ssp245['tas'].sel(time=slice('2071-01-01', '2100-12-31')).mean('time')
mean_2071_2100_ssp585 = dset_ssp585['tas'].sel(time=slice('2071-01-01', '2100-12-31')).mean('time')

diff_ssp119 = mean_2071_2100_ssp119 - mean_1850_1900
diff_ssp245 = mean_2071_2100_ssp245 - mean_1850_1900
diff_ssp585 = mean_2071_2100_ssp585 - mean_1850_1900


def plot_diffs_three_panel(diffs, titles, out_png):
    lon = diffs[0]['lon']
    lat = diffs[0]['lat']
    min_val = float(min(diff.min().values for diff in diffs))
    max_val = float(max(diff.max().values for diff in diffs))
    max_abs = max(abs(min_val), abs(max_val))
    vmin = -max_abs
    vmax = max_abs

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), sharex=True, sharey=True)
    meshes = []
    for ax, diff, title in zip(axes, diffs, titles):
        mesh = ax.pcolormesh(
            lon,
            lat,
            diff,
            cmap='coolwarm',
            shading='auto',
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel('Longitude (degrees)')
        ax.set_xlim(float(lon.min()), float(lon.max()))
        ax.set_ylim(float(lat.min()), float(lat.max()))
        meshes.append(mesh)

    axes[0].set_ylabel('Latitude (degrees)')
    fig.tight_layout(rect=[0, 0, 0.94, 1])
    cax = fig.add_axes([0.94, 0.18, 0.012, 0.64])
    cbar = fig.colorbar(meshes[0], cax=cax, label='Temperature change (K)')
    plt.savefig(out_png, dpi=300)
    plt.close(fig)


plot_diffs_three_panel(
    [diff_ssp119, diff_ssp245, diff_ssp585],
    ['SSP1-1.9', 'SSP2-4.5', 'SSP5-8.5'],
    'diff_ssp_three_panel.png',
)
