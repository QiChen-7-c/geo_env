import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

import tools

# Part 1: Load ISD data for Jeddah (2024)
df_isd = tools.read_isd_csv(
    r'/Users/chenq0d/Library/Mobile Documents/com~apple~CloudDocs/shared_doc/KAUST_course_material/Geo-Environmental Modeling & Analysis/Course_Data/ISD_Data/41024099999.csv'
)

plot = df_isd.plot(title="ISD data for Jeddah")
plt.savefig('isd_data_jeddah_2024.png', dpi=300)
plt.show()

# Part 2: Heat Index (HI) Calculation
# 1) Relative humidity from dewpoint and air temperature

df_isd['RH'] = tools.dewpoint_to_rh(df_isd['DEW'].values, df_isd['TMP'].values)

# 2) Heat index from air temperature and relative humidity

df_isd['HI'] = tools.gen_heat_index(df_isd['TMP'].values, df_isd['RH'].values)

# 3) Highest HI observed in the year
max_vals = df_isd.max(numeric_only=True)
max_hi = max_vals['HI']
print(f"Highest HI (C): {max_hi:.2f}")

# 4) Day and time (UTC) when highest HI was observed
max_idx = df_isd['HI'].idxmax()
print(f"Highest HI time (UTC): {max_idx}")

# 5) Local time in Jeddah (UTC+3)
max_idx_local = max_idx + pd.Timedelta(hours=3)
print(f"Highest HI time (Jeddah local, UTC+3): {max_idx_local}")

# 6) Air temperature and RH at this moment
row_at_max = df_isd.loc[max_idx]
print(f"Air temperature at max HI (C): {row_at_max['TMP']:.2f}")
print(f"Relative humidity at max HI (%): {row_at_max['RH']:.2f}")

# 7) NWS HI category and expected effects
hi_f = max_hi * 9 / 5 + 32
category = ""
expected_effects = ""
if hi_f < 80:
    category = "Below NWS thresholds"
    expected_effects = "Heat stress unlikely under normal conditions."
elif hi_f < 90:
    category = "Caution"
    expected_effects = "Fatigue possible with prolonged exposure and/or physical activity."
elif hi_f < 103:
    category = "Extreme Caution"
    expected_effects = "Heat stroke, heat cramps, or heat exhaustion possible with prolonged exposure and/or physical activity."
elif hi_f < 125:
    category = "Danger"
    expected_effects = "Heat cramps or heat exhaustion likely, and heat stroke possible with prolonged exposure and/or physical activity."
else:
    category = "Extreme Danger"
    expected_effects = "Heat stroke highly likely."

print(f"NWS category at max HI: {category}")
print(f"Expected effects: {expected_effects}")


# 10) Plot HI time series for the year
ax = df_isd['HI'].plot(title="Heat Index (HI) for Jeddah, 2024")
ax.set_ylabel("Heat Index (C)")
plt.tight_layout()
plt.savefig('hi_timeseries_2024.png', dpi=300)
plt.show()

# Part 3: Potential Impact of Climate Change
# 1) Projected warming at Jeddah under SSP2-4.5 (2071-2100 minus 1850-1900)
hist_1850_1949 = xr.open_dataset(
    r'/Users/chenq0d/Library/Mobile Documents/com~apple~CloudDocs/shared_doc/KAUST_course_material/Geo-Environmental Modeling & Analysis/Course_Data/Climate_Model_Data/tas_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_185001-194912.nc'
)
hist_1950_2014 = xr.open_dataset(
    r'/Users/chenq0d/Library/Mobile Documents/com~apple~CloudDocs/shared_doc/KAUST_course_material/Geo-Environmental Modeling & Analysis/Course_Data/Climate_Model_Data/tas_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_195001-201412.nc'
)
ssp245 = xr.open_dataset(
    r'/Users/chenq0d/Library/Mobile Documents/com~apple~CloudDocs/shared_doc/KAUST_course_material/Geo-Environmental Modeling & Analysis/Course_Data/Climate_Model_Data/tas_Amon_GFDL-ESM4_ssp245_r1i1p1f1_gr1_201501-210012.nc'
)

dset_hist = xr.concat(
    [hist_1850_1949, hist_1950_2014],
    dim='time',
    data_vars='minimal',
    coords='minimal',
    compat='override',
    join='override',
)
mean_1850_1900 = dset_hist['tas'].sel(time=slice('1850-01-01', '1900-12-31')).mean('time')
mean_2071_2100_ssp245 = ssp245['tas'].sel(time=slice('2071-01-01', '2100-12-31')).mean('time')
diff_ssp245 = mean_2071_2100_ssp245 - mean_1850_1900

lat_jeddah = 21.4858
lon_jeddah = 39.1925
if float(diff_ssp245['lon'].max()) > 180:
    lon_sel = lon_jeddah
else:
    lon_sel = lon_jeddah if lon_jeddah <= 180 else lon_jeddah - 360

warming_jeddah = float(diff_ssp245.sel(lat=lat_jeddah, lon=lon_sel, method='nearest').values)
print(f"Projected warming at Jeddah under SSP2-4.5 (C): {warming_jeddah:.2f}")

# 2) Apply warming to air temperature and recalculate HI
tmp_warm = df_isd['TMP'].values + warming_jeddah
rh_warm = tools.dewpoint_to_rh(df_isd['DEW'].values, tmp_warm)
hi_warm = tools.gen_heat_index(tmp_warm, rh_warm)

max_hi_warm = float(pd.Series(hi_warm).max())
increase_hi = max_hi_warm - max_hi
print(f"Highest HI with warming (C): {max_hi_warm:.2f}")
print(f"Increase in highest HI (C): {increase_hi:.2f}")
