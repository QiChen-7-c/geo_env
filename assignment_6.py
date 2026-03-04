from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import tools

matplotlib.use("Agg")

# ----------------------------
# Paths / settings
# ----------------------------

OUTPUT_DIR = Path(__file__).resolve().parent
DATA_PATH = OUTPUT_DIR / "download.nc"

# Target point (used only if the file contains a lat/lon grid)
lat_target = 21.25
lon_target = 39.5

# Reservoir area (given in assignment)
reservoir_area_km2 = 1.6

# ----------------------------
# Part 2: Data pre-processing
# ----------------------------

dset = xr.open_dataset(DATA_PATH)

# If the file contains a latitude/longitude grid, pick the closest gridpoint
if ("latitude" in dset.dims) and ("longitude" in dset.dims):
    dset = dset.sel(latitude=lat_target, longitude=lon_target, method="nearest")

# Some ERA5 downloads include preliminary/final data (expver dimension)
if "expver" in dset.dims:
    dset = dset.mean("expver", skipna=True)

lat_used = float(np.array(dset["latitude"]))
lon_used = float(np.array(dset["longitude"]))

time_name = None
if "time" in dset.coords:
    time_name = "time"
elif "valid_time" in dset.coords:
    time_name = "valid_time"
else:
    raise KeyError("Could not find a time coordinate (expected 'time' or 'valid_time').")

# Extract variables (numpy arrays)
t2m = np.array(dset.variables["t2m"]).astype("float64")
tp = np.array(dset.variables["tp"]).astype("float64")
time_dt = pd.to_datetime(np.array(dset[time_name]))

# Convert units
t2m = t2m - 273.15  # K -> °C
tp = tp * 1000.0  # m -> mm 

# Replace GRIB missing values (very large) with NaN
t2m[t2m > 1e20] = np.nan
tp[tp > 1e20] = np.nan

# Put into a dataframe
df_era5 = pd.DataFrame(index=time_dt)
df_era5["t2m"] = t2m
df_era5["tp"] = tp

print("Time range:", df_era5.index.min(), "to", df_era5.index.max())
print(f"Location: lat={lat_used:.2f}, lon={lon_used:.2f}")

# Plot time series
df_era5.plot()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "assignment_6_timeseries_t2m_tp.png", dpi=300)
plt.close()

# Average annual precipitation (mm/y): use annual total from hourly series
annual_tp_mm = df_era5["tp"].resample("YE").sum(min_count=1)
hours_per_year = df_era5["tp"].resample("YE").count()

# Drop incomplete years (simple rule)
full_year_mask = hours_per_year >= (360 * 24)
annual_tp_full_mm = annual_tp_mm[full_year_mask]
mean_annual_tp_mm = float(annual_tp_full_mm.mean())

print("\nAnnual precipitation totals (mm):")
print(annual_tp_mm)
print("\nMean annual precipitation over full years (mm/y):", mean_annual_tp_mm)

# Simple trend check using annual values (short record, interpret cautiously)
annual_tmean_c = df_era5["t2m"].resample("YE").mean()
years = annual_tmean_c.index.year.values
mask = full_year_mask.values & np.isfinite(annual_tmean_c.values) & np.isfinite(annual_tp_mm.values)

t_slope_c_per_year = float(np.polyfit(years[mask], annual_tmean_c.values[mask], 1)[0])
p_slope_mm_per_year2 = float(np.polyfit(years[mask], annual_tp_mm.values[mask], 1)[0])
print("\nTrend (annual mean temperature) slope (°C/year):", t_slope_c_per_year)
print("Trend (annual precipitation) slope (mm/year per year):", p_slope_mm_per_year2)

# Plot annual trends (line plots) for Part 2 Q9
annual_years = years[mask]
annual_t = annual_tmean_c.values[mask]
annual_p = annual_tp_mm.values[mask]

t_fit = np.polyval(np.polyfit(annual_years, annual_t, 1), annual_years)
p_fit = np.polyval(np.polyfit(annual_years, annual_p, 1), annual_years)

fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax[0].plot(annual_years, annual_t, marker="o", linewidth=1.2, label="Annual mean T")
ax[0].plot(annual_years, t_fit, linestyle="--", linewidth=1.2, label="Linear fit")
ax[0].set_ylabel("Temperature (°C)")
ax[0].grid(True, alpha=0.3)
ax[0].legend()

ax[1].plot(annual_years, annual_p, marker="o", linewidth=1.2, label="Annual P total")
ax[1].plot(annual_years, p_fit, linestyle="--", linewidth=1.2, label="Linear fit")
ax[1].set_ylabel("Precipitation (mm/year)")
ax[1].set_xlabel("Year")
ax[1].grid(True, alpha=0.3)
ax[1].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "assignment_6_trends_annual.png", dpi=300)
plt.close()

# ----------------------------
# Part 3: Potential evaporation
# ----------------------------

daily_tmin = df_era5["t2m"].resample("D").min()
daily_tmax = df_era5["t2m"].resample("D").max()
daily_tmean = df_era5["t2m"].resample("D").mean()

doy = daily_tmean.index.dayofyear.values

pe_mmd = tools.hargreaves_samani_1982(
    daily_tmin.values, daily_tmax.values, daily_tmean.values, lat_used, doy
)
pe_ts = pd.Series(pe_mmd, index=daily_tmean.index, name="pe_mmd")

# Plot PE
plt.figure(figsize=(10, 3.5))
plt.plot(pe_ts.index, pe_ts.values, linewidth=0.7, label="Potential evaporation (HS)")
plt.xlabel("Time")
plt.ylabel("Potential evaporation (mm/day)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "assignment_6_pe_daily.png", dpi=300)
plt.close()

# Mean annual PE (mm/y): annual totals from daily PE
annual_pe_mm = pe_ts.resample("YE").sum(min_count=1)
days_per_year = pe_ts.resample("YE").count()
full_year_mask_days = days_per_year >= 360
annual_pe_full_mm = annual_pe_mm[full_year_mask_days]
mean_annual_pe_mm = float(annual_pe_full_mm.mean())
print("\nMean annual PE over full years (mm/y):", mean_annual_pe_mm)

# Volume of water potentially lost (m3/y)
reservoir_area_m2 = reservoir_area_km2 * 1e6
vol_loss_m3y = mean_annual_pe_mm / 1000.0 * reservoir_area_m2
print("Potential annual evaporative volume loss (m3/y):", vol_loss_m3y)

 