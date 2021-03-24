#%%
# SARデータの外観を調べる
import os, glob

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

odir = "./outputs/radial_profiles"
os.makedirs(odir, exist_ok=True)
#%%
request_url = "https://cyclobs.ifremer.fr/app/api/getData?instrument=C-Band_SAR&include_cols=all"
data_list = pd.read_csv(request_url)
eye_list = data_list[data_list["eye_in_acq"]==True]
#%%
def parse_polygon(polygon):
    polygon = polygon.split("((")[1].split("))")[0].split(",")
    polygon = np.array(list(map(lambda x: np.array(x.strip().split(" ")).astype(float), polygon)))
    return polygon

def parse_point(point):
    point = point.split("(")[1].split(")")[0].split(" ")
    return point

def get_overview(oname):
    request_url = "https://cyclobs.ifremer.fr/app/api/getData?instrument=C-Band_SAR&include_cols=all"
    data_list = pd.read_csv(request_url)
    eye_list = data_list[data_list["eye_in_acq"]==True]

    lons, lats = [], []
    for track_point in eye_list["track_point"].values:
        lon, lat = parse_point(track_point)
        lons.append(lon)
        lats.append(lat)
    df = pd.DataFrame(np.array([eye_list["data_url"].apply(lambda x: os.path.basename(x)).values, eye_list["track_date"], lons, lats, eye_list["vmax (m/s)"], eye_list["mission"], eye_list["bounding_box"], eye_list["dist_eye_centroid"], eye_list["sid"], eye_list["cyclone_name"], eye_list["maximum_cyclone_category"]]).T, 
                    columns=["fname", "time", "lon", "lat", "vmax", "missionName", "bbox", "dist_eye_centroid", "sid", "cyclone_name", "maximum_cyclone_category"])
    df.to_csv(oname)

oname = "data/cyclobs_overview/overview.csv"
# get_overview(oname)

df = pd.read_csv(oname, parse_dates=["time"])
# %%
# SAR acquisition time
fig, ax = plt.subplots()
ax = df["time"].groupby(df["time"].dt.year).count().plot(kind="bar", width=0.7, color="#231f20")
ax.set(ylabel="Total number", axisbelow=True)
ax.set_xlabel(xlabel="SAR acquisition time (Year)", labelpad=5)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_color('#231f20')
ax.tick_params(left=False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(ls="--", alpha=0.5, axis="y")
fig.savefig(odir + "/SAR_acquisition_time.pdf", bbox_inches="tight", pad_inches=0.1)

# %%
fig, ax = plt.subplots()
ax = df["missionName"].groupby(df["missionName"]).count().plot(kind="bar", width=0.5, color="#231f20")
ax.set(ylabel="Total number", axisbelow=True)
ax.set_xlabel(xlabel="C-band SAR mission name", labelpad=5)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_color('#231f20')
ax.tick_params(left=False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(ls="--", alpha=0.5, axis="y")
fig.savefig(odir + "/SAR_missionName.pdf", bbox_inches="tight", pad_inches=0.1)

# %%
fig, ax = plt.subplots()
ax = pd.cut(df["vmax"], [0, 33, 43, 50, 58, 70, 80], labels=["Storm\n(<32)", "Cat.1\n(32-42)", "Cat.2\n(42-49)", "Cat.3\n(49-58)", "Cat.4\n(58-70)", "Cat.5\n(>70)"]).value_counts().sort_index().plot(kind="bar", width=0.6, color="#231f20")
ax.set(ylabel="Total number", axisbelow=True)
ax.set_xlabel(xlabel="Intensity (m/s)", labelpad=7)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_color('#231f20')
ax.tick_params(left=False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(ls="--", alpha=0.5, axis="y")
fig.savefig(odir + "/SAR_Vmax.pdf", bbox_inches="tight", pad_inches=0.1)

#%%
fig, ax = plt.subplots()
counts = df["maximum_cyclone_category"].groupby(df["maximum_cyclone_category"]).value_counts()
counts = pd.Series(data=[counts["storm"].item(), counts["cat-1"].item(), counts["cat-2"].item(), counts["cat-3"].item(), counts["cat-4"].item(), counts["cat-5"].item()], index=["Storm\n(<32)", "Cat.1\n(32-42)", "Cat.2\n(42-49)", "Cat.3\n(49-58)", "Cat.4\n(58-70)", "Cat.5\n(>70)"])
ax = counts.plot(kind="bar", width=0.6, color="#231f20")
ax.set(ylabel="Total number", axisbelow=True)
ax.set_xlabel(xlabel="Maximum intensity (m/s)", labelpad=7)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_color('#231f20')
ax.tick_params(left=False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(ls="--", alpha=0.5, axis="y")
fig.savefig(odir + "/SAR_maximum_intensity.pdf", bbox_inches="tight", pad_inches=0.1)

# %%
import cartopy.crs as ccrs
fig, ax = plt.subplots(figsize=(10,8), subplot_kw={'projection': ccrs.PlateCarree()})
cax = fig.add_axes([0.146, 0.38, 0.18, 0.025])
cax.set_title("Total number", fontsize=11, pad=0)

h = ax.hist2d(df["lon"], df["lat"], bins=[np.arange(-180,181,10),np.arange(-90,91,10)], vmin=0, cmap=plt.cm.get_cmap('Blues', 18))
ax.set(axisbelow=True, ylim=(-60,60))
ax.set_title("SAR acquisition position", loc="left")
ax.coastlines(color="g")
gl = ax.gridlines(draw_labels=True, linewidth=0.5, linestyle="--", alpha=0.5)
gl.xlabels_top = False
gl.ylabels_right = False
fig.colorbar(h[3], cax=cax, orientation="horizontal", pad=0.05, aspect=40, ticks=np.arange(0,30,3))
fig.savefig(odir + "/SAR_position.pdf", bbox_inches="tight", pad_inches=0.1)
# %%
import cartopy.crs as ccrs
fig, ax = plt.subplots(figsize=(10,8), subplot_kw={'projection': ccrs.PlateCarree()})
cax = fig.add_axes([0.146, 0.38, 0.18, 0.025])
cax.set_title("Total number", fontsize=11, pad=0)

h = ax.hist2d(df["lon"], df["lat"], bins=[np.arange(-180,181,10),np.arange(-90,91,10)], vmin=0, cmap=plt.cm.get_cmap('Blues', 18), alpha=1)

for footprint in df["bbox"].values:
    polygon = np.array(parse_polygon(footprint))
    if polygon[:,0].min() < -90 and polygon[:,0].max() > 90: # 180を跨ぐ事例
        polygon[:,0][polygon[:,0] < -90] += 360
        ax.plot(polygon[:,0], polygon[:,1], lw=0.8, alpha=0.8, c="k")
        ax.plot(polygon[:,0], polygon[:,1], lw=0.4, alpha=1, c="w")
        polygon[:,0] -= 360
        ax.plot(polygon[:,0], polygon[:,1], lw=0.8, alpha=0.8, c="k")
        ax.plot(polygon[:,0], polygon[:,1], lw=0.4, alpha=1, c="w")
    else:
        ax.plot(polygon[:,0], polygon[:,1], lw=0.8, alpha=0.8, c="k")
        ax.plot(polygon[:,0], polygon[:,1], lw=0.4, alpha=1, c="w")

ax.set(axisbelow=True, xlim=(-180,180), ylim=(-60,60))
ax.set_title("SAR acquisition position", loc="left")
ax.coastlines(color="g")
gl = ax.gridlines(draw_labels=True, linewidth=0.5, linestyle="--", alpha=0.5)
gl.xlabels_top = False
gl.ylabels_right = False
fig.colorbar(h[3], cax=cax, orientation="horizontal", pad=0.05, aspect=40, ticks=np.arange(0,30,3))
fig.savefig(odir + "/SAR_polygon.pdf", bbox_inches="tight", pad_inches=0.1)
# %%
