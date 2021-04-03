#%%
import os, glob

import cv2
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

from SAR import SAR

plt.rcParams["figure.dpi"] = 200
#%%
data_dir = "data/gridded/"
oname = "data/cyclobs_overview/overview.csv"
overview = pd.read_csv(oname, parse_dates=["time"])
overview = overview[overview["vmax"] >= 32] # Cat-1以上に限定

def center_estimation_Tsukada2021b(img, x1st, y1st, radius=60):
    ### triming
    img = np.pad(img, radius, mode="constant", constant_values=np.nan)
    x1st, y1st = x1st + radius, y1st + radius
    img = SAR.lee_filter(img[y1st-radius:y1st+radius, x1st-radius:x1st+radius], 3)
    ### eye extraction
    eye_region_candidates = img < 0.8 * np.nanmean(img) # high_wind_region = img > 1.2 * np.nanmean(img)

    lbls, nlbls = ndi.label(eye_region_candidates)
    sizes = np.bincount(lbls.ravel())[1:]

    img = np.where(np.isnan(img), 0, img)

    eyewall_features = {}
    for nlbl in range(1, nlbls+1):
        if sizes[nlbl-1] < size_min or sizes[nlbl-1] > size_max:
            continue
        lbl = (lbls == nlbl)
        dilated = ndi.binary_dilation(lbl, structure=np.ones([3,3]), iterations=15)
        eyewall = dilated ^ lbl
        eyewall_feature = ndi.median(img, eyewall) - ndi.minimum(img, lbl)
        if ~np.isnan(eyewall_feature):
            eyewall_features[nlbl] = eyewall_feature
    eye_lbl = max(eyewall_features, key=eyewall_features.get)
    y, x = ndi.center_of_mass(lbls == eye_lbl)
    x, y = x + x1st - 2 * radius, y + y1st - 2 * radius

    # rmax = int(rmax/(110*np.abs(sar.lat[0]-sar.lat[1])))
    # flags = cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LINEAR
    # wind_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # polar = cv2.warpPolar(wind_norm, (rmax, int(360/0.5)), (x, y), rmax, flags)
    
    # diff = np.diff(polar.astype(np.float), axis=1)
    # blurred_diff = ndi.gaussian_filter(diff, sigma=5)
    # eye_extents = np.argmax(blurred_diff, axis=1) # align with 0.5 deg
    # eye_region_polar = np.zeros_like(polar)

    # for i, eye_r in enumerate(eye_extents):
    #     eye_region_polar[i, :eye_r] = 1
    # eye_region = SAR.get_cart(eye_region_polar, rmax, rmax * 2, rmax * 2)
    # plt.imshow(eye_region)
    # plt.show()
    # ydiff, xdiff = ndi.measurements.center_of_mass(eye_region) - np.array([rmax, rmax])
    # x, y = x + xdiff, y + ydiff
    return x, y

odir = "outputs/sar_center_cat1-5"
os.makedirs(odir, exist_ok=True)
def plot_tc_center(sar, x=None, y=None, x1st=None, y1st=None, savedir=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig, ax = plt.subplots(figsize=(7,7), facecolor="white")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    ax.imshow(sar.wind_speed, vmin=0, vmax=80, cmap=SAR.get_wind_cmap())
    if x1st is not None and y1st is not None:
        ax.scatter(x1st, y1st, c="k", marker="+", linewidths=0.4, s=5)
    if x is not None and y is not None:
        ax.scatter(x, y, c="r", s=5)
    ax.set_title(f"SAR Derived Wind Speed\n{sar.time.dt.strftime('%Y-%m-%d %H:%M:%S').item()} UTC")
    ax.set(xlabel="X (px)", ylabel="Y (px)")
    ax.grid(alpha=0.3)
    fig.colorbar(ax.images[0], cax=cax)
    if savedir is None:
        plt.show()
    else:
        fig.savefig(savedir + f"/SAR_Center_{sar.time.dt.strftime('%Y-%m-%d_%Hh%Mm').item()}.png", bbox_inches="tight", pad_inches=0.1)
    plt.close()

#%%
size_min = 2 * 2 * np.pi
size_max = 60 * 60 * np.pi
radius = 60

x2nd, y2nd, lon2nd, lat2nd = [None]*len(overview), [None]*len(overview), [None]*len(overview), [None]*len(overview)
for i in range(0, len(overview)):
    ### data open
    tc_info = overview.iloc[i]
    fname = data_dir + "/" + tc_info["fname"]
    try:
        sar = xr.open_dataset(fname).isel(time=0)
    except:
        continue
    ### center definition
    lon1st, lat1st = tc_info["lon"], tc_info["lat"]
    x1st, y1st = np.argmin(np.abs(sar.lon.values-lon1st)), np.argmin(np.abs(sar.lat.values-lat1st))
    try:
        x, y = center_estimation_Tsukada2021b(sar.wind_speed.values, x1st, y1st, radius=50)
        # plot_tc_center(sar, x2nd, y2nd, x1st, y1st, savedir=odir)
        x2nd[i], y2nd[i] = x, y
        lon2nd[i] = sar.lon.assign_coords({"lon": np.arange(sar.lon.size)}).interp(lon=x).item()
        lat2nd[i] = sar.lat.assign_coords({"lat": np.arange(sar.lat.size)}).interp(lat=y).item()
    except:
        # plot_tc_center(sar, x1st=x1st, y1st=y1st, savedir=odir)
        import traceback
        traceback.print_exc()

# %%
overview["cx"] = x2nd
overview["cy"] = y2nd
overview["clon"] = lon2nd
overview["clat"] = lat2nd

overview.to_csv("data/cyclobs_overview/overview_with_center.csv")
# %%
