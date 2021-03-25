#%%
import os, glob

import cv2
import numpy as np
import pandas as pd
import xarray as xr
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 200
plt.rcParams["figure.facecolor"] = "white"

sar_wind_odir = "./outputs/SAR_wind_speed"
os.makedirs(sar_wind_odir, exist_ok=True)
radial_profile_odir = "./outputs/radial_profiles"
os.makedirs(radial_profile_odir, exist_ok=True)
#%%
# データオープン
data_dir = "data/gridded/"
fnames = sorted(glob.glob(data_dir+"*.nc"))

# overviewから緯度経度を取得
oname = "data/cyclobs_overview/overview.csv"
overview = pd.read_csv(oname, parse_dates=["time"])

def get_tc_info(overview, key, value):
    return overview[overview[key] == value].iloc[0]

def get_circle_label(width, height):
    yy, xx = np.mgrid[:height, :width]
    xx = np.abs(xx - width//2)
    yy = np.abs(yy - height//2)
    dist = np.sqrt(xx**2 + yy**2)
    dist = np.round(dist)
    return dist

def get_polar(img, radius, deg, px, py):
    flags = cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LINEAR
    polar = cv2.warpPolar(img, (radius, int(360/deg)), (px, py), radius, flags) 
    return polar

def get_4quad_winds(winds):
    winds_NE = winds.sel(a=slice(0,90)).isel(a=slice(0,-1))
    winds_NW = winds.sel(a=slice(90,180)).isel(a=slice(0,-1))
    winds_SE = winds.sel(a=slice(180,270)).isel(a=slice(0,-1))
    winds_SW = winds.sel(a=slice(270,360))
    return winds_NE, winds_NW, winds_SE, winds_SW

def plot_radial_profile(winds, sid, cyclone_name, quad="ALL", scatter=True, savedir="./"):
    winds_NE, winds_NW, winds_SE, winds_SW = get_4quad_winds(winds)
    winds_map = {"ALL": winds, "NE": winds_NE, "NW": winds_NW, "SE": winds_SE, "SW": winds_SW}

    winds = winds_map[quad.upper()]

    v_mean = np.nanmean(winds, axis=0)
    v_std = np.nanstd(winds, axis=0)
    v_max = np.nanmax(v_mean).item()
    r_max = winds.r[np.nanargmax(v_mean)].item()

    fig, ax = plt.subplots()

    if scatter:
        r_for_scatter = np.zeros(winds.a.size * winds.r.size)
        for i in range(len(winds.a)):
            r_for_scatter[i*winds.r.size:(i+1)*winds.r.size] = winds.r
        ax.scatter(r_for_scatter, winds.values.ravel(), c="darkgray", s=1, zorder=5)

    ax.plot(winds.r, v_mean, c="k", lw=1, zorder=10)
    ax.plot(winds.r, v_mean + v_std, ls="--", lw=0.7, c="k", dashes=(10, 4), zorder=9)
    ax.plot(winds.r, v_mean - v_std, ls="--", lw=0.7, c="k", dashes=(10, 4), zorder=9)
    ax.text(300, 75, f"Center lon: {round(winds.attrs['lon'],2)}°E", zorder=12)
    ax.text(300, 71, f"Center lat: {round(winds.attrs['lat'],2)}°N", zorder=12)
    ax.text(300, 65, f"Quadrant:   {quad.upper()}", zorder=12)
    ax.text(300, 61, f"Pixel Size:   {round(res_km)} km", zorder=12)
    ax.text(300, 55, r"$\mathsf{V_{max}}$"+f":    {round(v_max,1)} m/s", zorder=12)
    ax.text(300, 51, r"$\mathsf{R_{max}}$"+f":    {round(r_max,1)} km", zorder=12)
    ax.scatter(r_max, v_max, c="r", s=30, zorder=8)
    ax.plot([r_max, r_max], [0, v_max], c="r", lw=0.5, ls="--", zorder=8)
    ax.plot([0, r_max], [v_max, v_max], c="r", lw=0.5, ls="--", zorder=8)

    ax.set(xlim=(0,500), ylim=(0,80), axisbelow=True)
    ax.set_title(f"SAR Derived Wind Speed: {sid} / {cyclone_name}\n{winds.time.dt.strftime('%Y-%m-%d %H:%M:%S').item()} UTC")
    ax.set(xlabel="Distance from center (km)", ylabel="Wind speed (m/s)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)

    fig.savefig(savedir + f"/SAR_radial_profile_{winds.time.dt.strftime('%Y-%m-%d_%Hh').item()}_{tc_info['sid'].item()}_{tc_info['cyclone_name'].item()}_{quad.upper()}.png", bbox_inches="tight", pad_inches=0.1)
    plt.close()

from mpl_toolkits.axes_grid1 import make_axes_locatable
def plot_wind_speeds(sar, lon, lat, sid, cyclone_name, radius=0.5, res_km=1, savedir="."):
    """radius は lonlat で指定"""
    fig, ax = plt.subplots(figsize=(7,7))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)

    x, y = np.argmin(np.abs(sar.lon.values-lon)), np.argmin(np.abs(sar.lat.values-lat))

    ax.imshow(sar.wind_speed.values, extent=(sar.lon.min(), sar.lon.max(), sar.lat.min(), sar.lat.max()), vmin=0, vmax=80, cmap="jet")
    fig.colorbar(ax.images[0], cax=cax, shrink=0.85, pad=0.005)

    ax.grid(ls="-", c="w", lw=0.5, alpha=0.5)
    ax.set(xlim=(lon-radius, lon+radius), ylim=(lat-radius, lat+radius))
    ax.set_title(f"SAR Derived Wind Speed (m/s): {sid} / {cyclone_name}\n{sar.time.dt.strftime('%Y-%m-%d %H:%M:%S').item()} UTC")
    ax.set(xlabel="Longitude (°E)", ylabel="Latitude (°N)")
    fig.savefig(savedir + f"/SAR_winds_{winds.time.dt.strftime('%Y-%m-%d_%Hh').item()}_{tc_info['sid'].item()}_{tc_info['cyclone_name'].item()}_{quad.upper()}.png", bbox_inches="tight", pad_inches=0.1)
    plt.close()

#%%
max_radius = 500
res_km = 1
max_r_px = int(max_radius/res_km)

ddeg = 0.5
for fname in fnames:
    try:
        sar = xr.open_dataset(fname).isel(time=0)
        tc_info = get_tc_info(overview, "fname", os.path.basename(fname))
        lon, lat = tc_info["lon"], tc_info["lat"]
        x, y = np.argmin(np.abs(sar.lon.values-lon)), np.argmin(np.abs(sar.lat.values-lat))
        wind_speed = np.pad(sar.wind_speed.values, max_r_px, mode="constant", constant_values=np.nan)
        x, y = x+max_r_px, y+max_r_px
        
        wind_polar = get_polar(wind_speed, max_r_px, ddeg, x, y)
        wind_polar = wind_polar[::-1, :]
        radii = np.arange(0, max_radius, res_km)
        azimuth = np.arange(0, 360, ddeg)

        attrs = sar.wind_speed.attrs
        attrs.update({"lon":lon, "lat":lat})
        winds = xr.DataArray(wind_polar, {"a":azimuth, "r":radii, "time": sar.time}, dims=["a", "r"], name="wind_polar", attrs=attrs)
        winds["a"].attrs.update({"long_name":"azimuth from East", "standard_name": "azimuth", "units": "degrees"})
        winds["r"].attrs.update({"long_name":"radius", "standard_name": "radius", "units": "km"})
        
        sid = tc_info['sid'].item()
        name = tc_info['cyclone_name'].item()
        
        # SAR海上風を描く
        plot_wind_speeds(sar, lon, lat, sid, name, radius=1, savedir=sar_wind_odir)

        # Radial Profile を描く
        for quad in ["ALL", "NE", "NW", "SE", "SW"]:
            plot_radial_profile(winds, sid, name, quad=quad, scatter=True, savedir=radial_profile_odir)
    except:
        print("Some error has occured: " + fname)
        import traceback
        traceback.print_exc()

# %%
