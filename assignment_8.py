from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401
import scipy.optimize as opt
import xarray as xr


os.chdir(os.path.dirname(__file__))
print(os.getcwd())


WS_PATH = Path("data/WS_3/WS_3.shp")
SAUDI_PATH = Path("data/assignment7_shape/Saudi_Shape.shp")

STUDY_AREA_PNG = Path("assignment_8_study_area.png")
PART1_PNG = Path("assignment_8_hourly_hydro_2001_2002.png")
PART2_TS_PNG = Path("assignment_8_part2_validation_2001_timeseries.png")
PART2_SCATTER_PNG = Path("assignment_8_part2_validation_2001_scatter.png")
CAL_TS_PNG = Path("assignment_8_part3_calibration_2001_timeseries.png")
CAL_SCATTER_PNG = Path("assignment_8_part3_calibration_2001_scatter.png")
VAL_TS_PNG = Path("assignment_8_part3_validation_2002_timeseries.png")
VAL_SCATTER_PNG = Path("assignment_8_part3_validation_2002_scatter.png")


def load_and_clip(nc_file: str | Path, var_name: str, gdf: gpd.GeoDataFrame) -> xr.DataArray:
    ds = xr.open_dataset(nc_file)
    da = ds[var_name]
    if "expver" in da.dims:
        da = da.mean("expver", skipna=True)
    da = da.astype("float64")
    da = da.rio.write_crs("EPSG:4326")
    da = da.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
    clipped = da.rio.clip(gdf.geometry, gdf.crs, drop=True).load()
    ds.close()
    return clipped


def simulate_runoff(k: float, P: np.ndarray, ET: np.ndarray, Q0: float, dt: float = 1.0) -> np.ndarray:
    n = len(P)
    Q_sim = np.zeros(n)
    Q_sim[0] = Q0
    for t in range(1, n):
        Q_t = (Q_sim[t - 1] + (P[t] - ET[t]) * dt) / (1 + dt / k)
        Q_sim[t] = max(0, Q_t)
    return Q_sim


def kge(Q_obs: np.ndarray, Q_sim: np.ndarray) -> tuple[float, float, float, float]:
    r = np.corrcoef(Q_obs, Q_sim)[0, 1]
    alpha = np.std(Q_sim) / np.std(Q_obs)
    beta = np.mean(Q_sim) / np.mean(Q_obs)
    KGE = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return float(KGE), float(r), float(alpha), float(beta)


def save_runoff_plots(time_index: pd.Index, Q_obs: np.ndarray, Q_sim: np.ndarray, title: str, ts_file: Path, scatter_file: Path) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(time_index, Q_obs, label="Observed/ERA5 runoff", color="#2ca02c", linewidth=0.9)
    plt.plot(time_index, Q_sim, label="Simulated runoff", color="#1f77b4", linewidth=0.9)
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Runoff (mm/h)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ts_file, dpi=300)
    plt.close()

    plt.figure(figsize=(6.2, 6))
    plt.scatter(Q_obs, Q_sim, s=10, alpha=0.35, color="#1f77b4", edgecolors="none")
    max_val = max(np.max(Q_obs), np.max(Q_sim))
    plt.plot([0, max_val], [0, max_val], "r--", linewidth=1.1)
    plt.xlabel("Observed runoff (mm/h)")
    plt.ylabel("Simulated runoff (mm/h)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(scatter_file, dpi=300)
    plt.close()


def plot_study_area(ws_gdf: gpd.GeoDataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 7))
    if SAUDI_PATH.exists():
        saudi_gdf = gpd.read_file(SAUDI_PATH)
        saudi_gdf.plot(ax=ax, facecolor="#f2efe8", edgecolor="#6b645c", linewidth=0.8)
    ws_gdf.plot(ax=ax, facecolor="#d95f02", edgecolor="black", linewidth=1.0)
    minx, miny, maxx, maxy = ws_gdf.total_bounds
    ax.set_xlim(minx - 1.8, maxx + 1.8)
    ax.set_ylim(miny - 1.5, maxy + 1.5)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Selected watershed (WS_3) in south-west Saudi Arabia")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(STUDY_AREA_PNG, dpi=300)
    plt.close(fig)


gdf = gpd.read_file(WS_PATH)
plot_study_area(gdf)


# -------- Load 2001 ERA5 datasets --------
precip_file = "data/assignment7_data/Precipitation/era5_OLR_2001_total_precipitation.nc"
et_file = "data/assignment7_data/Total_Evaporation/era5_OLR_2001_total_evaporation.nc"
runoff_file = "data/assignment7_data/Runoff/ambientera5_OLR_2001_total_runoff.nc"

P_grid = load_and_clip(precip_file, "tp", gdf) * 1000
ET_grid = load_and_clip(et_file, "e", gdf) * 1000
Q_grid = load_and_clip(runoff_file, "ro", gdf) * 1000

time_name = "valid_time" if "valid_time" in P_grid.coords else "time"
time_2001 = pd.to_datetime(P_grid[time_name].values)
P = P_grid.mean(dim=["latitude", "longitude"], skipna=True).values
ET = ET_grid.mean(dim=["latitude", "longitude"], skipna=True).values
Q_obs = Q_grid.mean(dim=["latitude", "longitude"], skipna=True).values
ET = np.where(ET < 0, -ET, ET)


# -------- Load 2002 ERA5 datasets --------
precip_file_val = "data/assignment7_data/Precipitation/era5_OLR_2002_total_precipitation.nc"
et_file_val = "data/assignment7_data/Total_Evaporation/era5_OLR_2002_total_evaporation.nc"
runoff_file_val = "data/assignment7_data/Runoff/ambientera5_OLR_2002_total_runoff.nc"

P_grid_val = load_and_clip(precip_file_val, "tp", gdf) * 1000
ET_grid_val = load_and_clip(et_file_val, "e", gdf) * 1000
Q_grid_val = load_and_clip(runoff_file_val, "ro", gdf) * 1000

time_name_val = "valid_time" if "valid_time" in P_grid_val.coords else "time"
time_2002 = pd.to_datetime(P_grid_val[time_name_val].values)
P_v = P_grid_val.mean(dim=["latitude", "longitude"], skipna=True).values
ET_v = ET_grid_val.mean(dim=["latitude", "longitude"], skipna=True).values
Q_obs_v = Q_grid_val.mean(dim=["latitude", "longitude"], skipna=True).values
ET_v = np.where(ET_v < 0, -ET_v, ET_v)

print("Part 1 preprocessing completed")


# -------- Plot variables --------
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=False)

axes[0].plot(time_2001, P, label="Precipitation", color="#1f77b4", linewidth=0.8)
axes[0].plot(time_2001, ET, label="Evaporation", color="#d62728", linewidth=0.8)
axes[0].plot(time_2001, Q_obs, label="Runoff", color="#2ca02c", linewidth=0.8)
axes[0].legend()
axes[0].set_xlabel("Time")
axes[0].set_ylabel("mm/h")
axes[0].set_title("Hydrological Variables over WS_3 (2001)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(time_2002, P_v, label="Precipitation", color="#1f77b4", linewidth=0.8)
axes[1].plot(time_2002, ET_v, label="Evaporation", color="#d62728", linewidth=0.8)
axes[1].plot(time_2002, Q_obs_v, label="Runoff", color="#2ca02c", linewidth=0.8)
axes[1].legend()
axes[1].set_xlabel("Time")
axes[1].set_ylabel("mm/h")
axes[1].set_title("Hydrological Variables over WS_3 (2002)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PART1_PNG, dpi=300)
plt.close()


# -------- Validate model with given k --------
k_test = 0.15
Q_sim = simulate_runoff(k_test, P, ET, Q_obs[0])
KGE, r, alpha, beta = kge(Q_obs, Q_sim)

print("\nValidation Results (2001, fixed k = 0.15)")
print("KGE:", KGE)
print("Correlation:", r)
print("Alpha:", alpha)
print("Beta:", beta)

save_runoff_plots(
    time_2001,
    Q_obs,
    Q_sim,
    "Observed vs Simulated Runoff (Validation 2001, k = 0.15)",
    PART2_TS_PNG,
    PART2_SCATTER_PNG,
)


# -------- Objective function for optimization --------
def objective(k: float, P: np.ndarray, ET: np.ndarray, Q_obs: np.ndarray) -> float:
    Q_sim = simulate_runoff(k, P, ET, Q_obs[0])
    return 1 - kge(Q_obs, Q_sim)[0]


res = opt.minimize_scalar(
    objective,
    bounds=(0.1, 2),
    args=(P, ET, Q_obs),
    method="bounded",
)
best_k = float(res.x)

print("\nOptimized k:", best_k)

Q_sim_cal = simulate_runoff(best_k, P, ET, Q_obs[0])
KGE_cal, r_cal, alpha_cal, beta_cal = kge(Q_obs, Q_sim_cal)

print("\nCalibration Results (2001)")
print("KGE:", KGE_cal)
print("r:", r_cal)
print("alpha:", alpha_cal)
print("beta:", beta_cal)

save_runoff_plots(
    time_2001,
    Q_obs,
    Q_sim_cal,
    f"Observed vs Simulated Runoff (Calibration 2001, k = {best_k:.4f})",
    CAL_TS_PNG,
    CAL_SCATTER_PNG,
)


# -------- Validation for 2002 using calibrated k --------
Q_sim_v = simulate_runoff(best_k, P_v, ET_v, Q_obs_v[0])
KGE_v, r_v, alpha_v, beta_v = kge(Q_obs_v, Q_sim_v)

print("\nValidation Results (2002)")
print("KGE:", KGE_v)
print("r:", r_v)
print("alpha:", alpha_v)
print("beta:", beta_v)

save_runoff_plots(
    time_2002,
    Q_obs_v,
    Q_sim_v,
    f"Observed vs Simulated Runoff (Validation 2002, k = {best_k:.4f})",
    VAL_TS_PNG,
    VAL_SCATTER_PNG,
)


print("\nGenerated figures:")
for path in [
    STUDY_AREA_PNG,
    PART1_PNG,
    PART2_TS_PNG,
    PART2_SCATTER_PNG,
    CAL_TS_PNG,
    CAL_SCATTER_PNG,
    VAL_TS_PNG,
    VAL_SCATTER_PNG,
]:
    print("-", path.name)
