from __future__ import annotations

import calendar
import re
import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.path import Path as MplPath


OUTPUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUTPUT_DIR / "data" / "assignment7_data"
SHAPEFILE_PATH = OUTPUT_DIR / "data" / "assignment7_shape" / "Saudi_Shape.shp"

PRECIP_DIR = DATA_DIR / "Precipitation"
EVAP_DIR = DATA_DIR / "Total_Evaporation"
RUNOFF_DIR = DATA_DIR / "Runoff"

START_YEAR = 2000
END_YEAR = 2020

MONTHLY_CSV = OUTPUT_DIR / "assignment_7_monthly_water_balance.csv"
ANNUAL_CSV = OUTPUT_DIR / "assignment_7_annual_water_balance.csv"
MONTHLY_PNG = OUTPUT_DIR / "assignment_7_monthly_water_balance.png"
CLIM_PNG = OUTPUT_DIR / "assignment_7_climatology.png"
ANNUAL_PNG = OUTPUT_DIR / "assignment_7_annual_trends.png"
SCATTER_PNG = OUTPUT_DIR / "assignment_7_runoff_vs_precip.png"
RUNOFF_PE_PNG = OUTPUT_DIR / "assignment_7_runoff_vs_p_minus_e.png"
ANSWERS_MD = OUTPUT_DIR / "assignment_7_part2_3_answers.md"


# Read the Saudi polygon from the shapefile directly.
rings = []
with SHAPEFILE_PATH.open("rb") as shp_file:
    shp_file.read(100)
    while True:
        record_header = shp_file.read(8)
        if not record_header:
            break

        _, record_length_words = struct.unpack(">2i", record_header)
        record = shp_file.read(record_length_words * 2)
        shape_type = struct.unpack("<i", record[:4])[0]
        if shape_type != 5:
            continue

        num_parts, num_points = struct.unpack("<2i", record[36:44])
        parts = struct.unpack("<" + "i" * num_parts, record[44 : 44 + 4 * num_parts])
        points_offset = 44 + 4 * num_parts
        points = np.frombuffer(
            record[points_offset : points_offset + 16 * num_points],
            dtype="<f8",
        ).reshape(-1, 2)

        part_limits = list(parts) + [num_points]
        for start, end in zip(part_limits[:-1], part_limits[1:]):
            ring = points[start:end]
            if len(ring) >= 3:
                rings.append(ring)

if not rings:
    raise RuntimeError(f"No polygon rings were found in {SHAPEFILE_PATH}.")


# Use the precipitation grid as the reference grid.
sample_precip_path = sorted(PRECIP_DIR.glob("*.nc"))[0]
sample_ds = xr.open_dataset(sample_precip_path)
latitudes = sample_ds["latitude"].values
longitudes = sample_ds["longitude"].values
sample_ds.close()

lon_2d, lat_2d = np.meshgrid(longitudes, latitudes)
grid_points = np.column_stack([lon_2d.ravel(), lat_2d.ravel()])
country_mask_flat = np.zeros(grid_points.shape[0], dtype=bool)

for ring in rings:
    country_mask_flat ^= MplPath(ring).contains_points(grid_points, radius=1e-9)

country_mask = country_mask_flat.reshape(lat_2d.shape)
if not country_mask.any():
    raise RuntimeError("Saudi mask is empty. Check the shapefile and grid.")

cell_weights = np.cos(np.deg2rad(latitudes))[:, None] * country_mask


# Map each year to its file.
precip_files = {}
for path in sorted(PRECIP_DIR.glob("*.nc")):
    match = re.search(r"_(\d{4})_", path.name)
    if match:
        precip_files[int(match.group(1))] = path

evap_files = {}
for path in sorted(EVAP_DIR.glob("*.nc")):
    match = re.search(r"_(\d{4})_", path.name)
    if match:
        evap_files[int(match.group(1))] = path

runoff_files = {}
for path in sorted(RUNOFF_DIR.glob("*.nc")):
    match = re.search(r"_(\d{4})_", path.name)
    if match:
        runoff_files[int(match.group(1))] = path

expected_years = set(range(START_YEAR, END_YEAR + 1))
if set(precip_files) != expected_years:
    missing = sorted(expected_years - set(precip_files))
    raise FileNotFoundError(f"Missing precipitation files for years: {missing}")
if set(evap_files) != expected_years:
    missing = sorted(expected_years - set(evap_files))
    raise FileNotFoundError(f"Missing evaporation files for years: {missing}")
if set(runoff_files) != expected_years:
    missing = sorted(expected_years - set(runoff_files))
    raise FileNotFoundError(f"Missing runoff files for years: {missing}")


# Read each yearly file and convert it to a Saudi-area monthly mean depth.
monthly_precip_parts = []
for year in range(START_YEAR, END_YEAR + 1):
    ds = xr.open_dataset(precip_files[year])
    time_name = "valid_time" if "valid_time" in ds.coords else "time"
    da = ds["tp"].astype("float64")
    if "expver" in da.dims:
        da = da.mean("expver", skipna=True)
    da = da.where(np.isfinite(da))
    da = da.where(np.abs(da) < 1e20)
    da = da * 1000.0
    monthly = da.resample({time_name: "MS"}).sum(skipna=True)
    monthly_values = monthly.values
    numerator = np.nansum(monthly_values * cell_weights[None, :, :], axis=(1, 2))
    denominator = np.nansum(
        np.where(np.isfinite(monthly_values), cell_weights[None, :, :], 0.0),
        axis=(1, 2),
    )
    monthly_precip_parts.append(
        pd.Series(numerator / denominator, index=pd.to_datetime(monthly[time_name].values))
    )
    ds.close()

monthly_evap_parts = []
for year in range(START_YEAR, END_YEAR + 1):
    ds = xr.open_dataset(evap_files[year])
    time_name = "valid_time" if "valid_time" in ds.coords else "time"
    da = ds["e"].astype("float64")
    if "expver" in da.dims:
        da = da.mean("expver", skipna=True)
    da = da.where(np.isfinite(da))
    da = da.where(np.abs(da) < 1e20)
    da = da * -1000.0
    monthly = da.resample({time_name: "MS"}).sum(skipna=True)
    monthly_values = monthly.values
    numerator = np.nansum(monthly_values * cell_weights[None, :, :], axis=(1, 2))
    denominator = np.nansum(
        np.where(np.isfinite(monthly_values), cell_weights[None, :, :], 0.0),
        axis=(1, 2),
    )
    monthly_evap_parts.append(
        pd.Series(numerator / denominator, index=pd.to_datetime(monthly[time_name].values))
    )
    ds.close()

monthly_runoff_parts = []
for year in range(START_YEAR, END_YEAR + 1):
    ds = xr.open_dataset(runoff_files[year])
    time_name = "valid_time" if "valid_time" in ds.coords else "time"
    da = ds["ro"].astype("float64")
    if "expver" in da.dims:
        da = da.mean("expver", skipna=True)
    da = da.where(np.isfinite(da))
    da = da.where(np.abs(da) < 1e20)
    da = da * 1000.0
    monthly = da.resample({time_name: "MS"}).sum(skipna=True)
    monthly_values = monthly.values
    numerator = np.nansum(monthly_values * cell_weights[None, :, :], axis=(1, 2))
    denominator = np.nansum(
        np.where(np.isfinite(monthly_values), cell_weights[None, :, :], 0.0),
        axis=(1, 2),
    )
    monthly_runoff_parts.append(
        pd.Series(numerator / denominator, index=pd.to_datetime(monthly[time_name].values))
    )
    ds.close()


# Combine everything in one table.
monthly_df = pd.DataFrame(
    {
        "precip_mm": pd.concat(monthly_precip_parts).sort_index(),
        "evap_mm": pd.concat(monthly_evap_parts).sort_index(),
        "runoff_mm": pd.concat(monthly_runoff_parts).sort_index(),
    }
)
monthly_df.index.name = "month"
monthly_df["balance_mm"] = monthly_df["precip_mm"] - (
    monthly_df["evap_mm"] + monthly_df["runoff_mm"]
)
monthly_df["p_minus_e_mm"] = monthly_df["precip_mm"] - monthly_df["evap_mm"]

annual_df = monthly_df.resample("YE").sum()
annual_df.index = annual_df.index.year
annual_df.index.name = "year"

climatology_df = monthly_df.groupby(monthly_df.index.month).mean(numeric_only=True)
climatology_df.index.name = "month"

monthly_df.to_csv(MONTHLY_CSV, float_format="%.6f")
annual_df.to_csv(ANNUAL_CSV, float_format="%.6f")


# Plot the monthly time series and the residual.
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axes[0].plot(monthly_df.index, monthly_df["precip_mm"], color="#1f77b4", linewidth=1.2, label="Precipitation")
axes[0].plot(monthly_df.index, monthly_df["evap_mm"], color="#d62728", linewidth=1.2, label="Evaporation")
axes[0].plot(monthly_df.index, monthly_df["runoff_mm"], color="#2ca02c", linewidth=1.2, label="Runoff")
axes[0].set_ylabel("Monthly total (mm)")
axes[0].set_title("Saudi Arabia monthly water-balance components (2000-2020)")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].axhline(0.0, color="black", linewidth=1.0, linestyle="--")
axes[1].plot(monthly_df.index, monthly_df["balance_mm"], color="#4c4c4c", linewidth=1.2, label="P - (E + R)")
axes[1].fill_between(
    monthly_df.index,
    0.0,
    monthly_df["balance_mm"],
    where=monthly_df["balance_mm"] >= 0.0,
    color="#2ca02c",
    alpha=0.25,
    interpolate=True,
)
axes[1].fill_between(
    monthly_df.index,
    0.0,
    monthly_df["balance_mm"],
    where=monthly_df["balance_mm"] < 0.0,
    color="#d62728",
    alpha=0.2,
    interpolate=True,
)
axes[1].set_ylabel("Monthly total (mm)")
axes[1].set_xlabel("Time")
axes[1].set_title("Residual storage term")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig(MONTHLY_PNG, dpi=300)
plt.close(fig)


# Plot the climatological seasonal cycle.
month_numbers = climatology_df.index.to_numpy(dtype=int)
month_labels = [calendar.month_abbr[month] for month in month_numbers]

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

axes[0].plot(month_numbers, climatology_df["precip_mm"], marker="o", linewidth=1.6, color="#1f77b4", label="Precipitation")
axes[0].plot(month_numbers, climatology_df["evap_mm"], marker="o", linewidth=1.6, color="#d62728", label="Evaporation")
axes[0].plot(month_numbers, climatology_df["runoff_mm"], marker="o", linewidth=1.6, color="#2ca02c", label="Runoff")
axes[0].set_ylabel("Climatological monthly total (mm)")
axes[0].set_title("Mean seasonal cycle over 2000-2020")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].axhline(0.0, color="black", linewidth=1.0, linestyle="--")
axes[1].plot(month_numbers, climatology_df["balance_mm"], marker="o", linewidth=1.6, color="#4c4c4c", label="P - (E + R)")
axes[1].set_ylabel("Climatological monthly total (mm)")
axes[1].set_xlabel("Month")
axes[1].set_title("Seasonal cycle of the water-balance residual")
axes[1].set_xticks(month_numbers)
axes[1].set_xticklabels(month_labels)
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig(CLIM_PNG, dpi=300)
plt.close(fig)


# Plot annual totals with linear trends.
annual_years = annual_df.index.to_numpy(dtype=float)

precip_slope, precip_intercept = np.polyfit(annual_years, annual_df["precip_mm"], 1)
evap_slope, evap_intercept = np.polyfit(annual_years, annual_df["evap_mm"], 1)
runoff_slope, runoff_intercept = np.polyfit(annual_years, annual_df["runoff_mm"], 1)
balance_slope, balance_intercept = np.polyfit(annual_years, annual_df["balance_mm"], 1)

precip_fit = precip_slope * annual_years + precip_intercept
evap_fit = evap_slope * annual_years + evap_intercept
runoff_fit = runoff_slope * annual_years + runoff_intercept
balance_fit = balance_slope * annual_years + balance_intercept

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axes[0].plot(
    annual_years,
    annual_df["precip_mm"],
    marker="o",
    linewidth=1.3,
    color="#1f77b4",
    label=f"Precipitation (trend {precip_slope * 10:.2f} mm/decade)",
)
axes[0].plot(annual_years, precip_fit, color="#1f77b4", linewidth=1.0, linestyle="--")

axes[0].plot(
    annual_years,
    annual_df["evap_mm"],
    marker="o",
    linewidth=1.3,
    color="#d62728",
    label=f"Evaporation (trend {evap_slope * 10:.2f} mm/decade)",
)
axes[0].plot(annual_years, evap_fit, color="#d62728", linewidth=1.0, linestyle="--")

axes[0].plot(
    annual_years,
    annual_df["runoff_mm"],
    marker="o",
    linewidth=1.3,
    color="#2ca02c",
    label=f"Runoff (trend {runoff_slope * 10:.2f} mm/decade)",
)
axes[0].plot(annual_years, runoff_fit, color="#2ca02c", linewidth=1.0, linestyle="--")

axes[0].set_ylabel("Annual total (mm/year)")
axes[0].set_title("Annual water-balance components and linear trends")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].axhline(0.0, color="black", linewidth=1.0, linestyle="--")
axes[1].plot(
    annual_years,
    annual_df["balance_mm"],
    marker="o",
    linewidth=1.3,
    color="#4c4c4c",
    label=f"P - (E + R) (trend {balance_slope * 10:.2f} mm/decade)",
)
axes[1].plot(annual_years, balance_fit, color="#4c4c4c", linewidth=1.0, linestyle="--")
axes[1].set_ylabel("Annual total (mm/year)")
axes[1].set_xlabel("Year")
axes[1].set_title("Annual residual water balance")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig(ANNUAL_PNG, dpi=300)
plt.close(fig)


# Plot runoff against precipitation.
monthly_corr_pr = float(monthly_df["precip_mm"].corr(monthly_df["runoff_mm"]))
scatter_slope, scatter_intercept = np.polyfit(monthly_df["precip_mm"], monthly_df["runoff_mm"], 1)
x_fit = np.linspace(0.0, float(monthly_df["precip_mm"].max()) * 1.05, 100)
y_fit = scatter_slope * x_fit + scatter_intercept

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(
    monthly_df["precip_mm"],
    monthly_df["runoff_mm"],
    s=22,
    alpha=0.75,
    color="#1f77b4",
    edgecolors="none",
)
ax.plot(x_fit, y_fit, color="#d62728", linewidth=1.5)
ax.set_xlabel("Monthly precipitation (mm)")
ax.set_ylabel("Monthly runoff (mm)")
ax.set_title(f"Runoff vs precipitation (monthly Pearson r = {monthly_corr_pr:.2f})")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(SCATTER_PNG, dpi=300)
plt.close(fig)


# Plot runoff against P - E to show their relative size and weak relationship.
monthly_corr_pe_r = float(monthly_df["p_minus_e_mm"].corr(monthly_df["runoff_mm"]))
pe_scatter_slope, pe_scatter_intercept = np.polyfit(monthly_df["p_minus_e_mm"], monthly_df["runoff_mm"], 1)
pe_x_fit = np.linspace(float(monthly_df["p_minus_e_mm"].min()) * 1.05, float(monthly_df["p_minus_e_mm"].max()) * 1.05, 100)
pe_y_fit = pe_scatter_slope * pe_x_fit + pe_scatter_intercept

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

axes[0].axhline(0.0, color="black", linewidth=1.0, linestyle="--")
axes[0].plot(
    annual_df.index,
    annual_df["p_minus_e_mm"],
    marker="o",
    linewidth=1.4,
    color="#4c4c4c",
    label="Annual P - E",
)
axes[0].set_ylabel("P - E (mm/year)", color="#4c4c4c")
axes[0].tick_params(axis="y", labelcolor="#4c4c4c")
axes[0].grid(True, alpha=0.3)

ax2 = axes[0].twinx()
ax2.plot(
    annual_df.index,
    annual_df["runoff_mm"],
    marker="s",
    linewidth=1.2,
    color="#2ca02c",
    label="Annual runoff",
)
ax2.set_ylabel("Runoff (mm/year)", color="#2ca02c")
ax2.tick_params(axis="y", labelcolor="#2ca02c")
axes[0].set_title("Annual comparison of P - E and runoff")

lines1, labels1 = axes[0].get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
axes[0].legend(lines1 + lines2, labels1 + labels2, loc="upper left")

axes[1].scatter(
    monthly_df["p_minus_e_mm"],
    monthly_df["runoff_mm"],
    s=22,
    alpha=0.75,
    color="#1f77b4",
    edgecolors="none",
)
axes[1].plot(pe_x_fit, pe_y_fit, color="#d62728", linewidth=1.5)
axes[1].axvline(0.0, color="black", linewidth=1.0, linestyle="--")
axes[1].set_xlabel("Monthly P - E (mm)")
axes[1].set_ylabel("Monthly runoff (mm)")
axes[1].set_title(f"Monthly runoff vs P - E (Pearson r = {monthly_corr_pe_r:.2f})")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RUNOFF_PE_PNG, dpi=300)
plt.close(fig)


# Prepare the text answers.
mean_annual = annual_df.mean()
annual_corr_pr = float(annual_df["precip_mm"].corr(annual_df["runoff_mm"]))

positive_months = monthly_df[monthly_df["balance_mm"] > 0.0]
positive_counts = positive_months.groupby(positive_months.index.month).size()
positive_month_names = ", ".join(calendar.month_name[month] for month in positive_counts.index)

wettest_year = int(annual_df["precip_mm"].idxmax())
driest_year = int(annual_df["precip_mm"].idxmin())
highest_balance_year = int(annual_df["balance_mm"].idxmax())
lowest_balance_year = int(annual_df["balance_mm"].idxmin())

highest_balance_month = monthly_df["balance_mm"].idxmax().strftime("%Y-%m")
lowest_balance_month = monthly_df["balance_mm"].idxmin().strftime("%Y-%m")

positive_climatology = climatology_df[climatology_df["balance_mm"] > 0.0]
climatology_months = ", ".join(calendar.month_name[int(month)] for month in positive_climatology.index)



print("Generated:")
print(f"- {MONTHLY_CSV.name}")
print(f"- {ANNUAL_CSV.name}")
print(f"- {MONTHLY_PNG.name}")
print(f"- {CLIM_PNG.name}")
print(f"- {ANNUAL_PNG.name}")
print(f"- {SCATTER_PNG.name}")
print(f"- {RUNOFF_PE_PNG.name}")


print("\nMean annual totals (mm/year):")
print(mean_annual.round(3))

print("\nTrend slopes (mm/year per year):")
print(
    pd.Series(
        {
            "precip_mm": precip_slope,
            "evap_mm": evap_slope,
            "runoff_mm": runoff_slope,
            "balance_mm": balance_slope,
        }
    ).round(4)
)

print("\nMonthly precipitation-runoff correlation:")
print(f"{monthly_corr_pr:.3f}")
