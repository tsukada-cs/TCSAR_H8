import cv2
import numpy as np
import pandas as pd
import xarray as xr
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

from sympy.geometry import Polygon
class SAR:
    @staticmethod
    def get_wind_cmap():
        from matplotlib.colors import LinearSegmentedColormap
        colors = (
            (0 / 80, "#ac00fe"),
            (8 / 80, "#5521ff"),
            (16 / 80, "#0fe8ff"),
            (24 / 80, "#30ff6c"),
            (32 / 80, "#fffd00"),
            (40 / 80, "#ffd700"),
            (48 / 80, "#fe8801"),
            (56 / 80, "#ec0100"),
            (64 / 80, "#cd1e6d"),
            (70 / 80, "#f53de2"),
            (80 / 80, "#ffffff")
        )
        cmap = LinearSegmentedColormap.from_list("SAR_wind", colors)
        return cmap
        
    @staticmethod
    def get_polygon(sar):
        footprint = sar.attrs["footprint"]
        polygon = footprint.split("((")[1].split("))")[0].split(",")
        polygon = list(map(lambda x: np.array(x.strip().split(" ")).astype(float), polygon))
        return Polygon(*polygon)

    @staticmethod
    def lee_filter(img, size):
        from scipy.ndimage.filters import uniform_filter
        from scipy.ndimage.measurements import variance
        mask = np.isnan(img)
        img[mask] = 0.0
        img_mean = uniform_filter(img, (size, size))
        img_sqr_mean = uniform_filter(img**2, (size, size))
        img_variance = img_sqr_mean - img_mean**2

        overall_variance = variance(img)

        img_weights = img_variance / (img_variance + overall_variance)
        img_output = img_mean + img_weights * (img - img_mean)
        img_output[mask] = np.nan
        return img_output

    @staticmethod
    def get_polar(img, radius, deg, px, py):
        flags = cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LINEAR
        norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        polar = cv2.warpPolar(norm, (radius, int(360/deg)), (px, py), radius, flags) 
        return polar

    @staticmethod
    def get_cart(img, radius, width, height):
        flags = cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP
        cart = cv2.warpPolar(img, (width, height), (width//2, height//2), radius, flags)
        return cart
    
    @staticmethod
    def center_estimation_Xu2016(sar, lon1st, lat1st, rect_radius=60, dist_thresh=0.2, polar_radius=100, polar_deg=0.5, max_loop=100):
        import scipy.ndimage as ndi
        nrcs = np.where(np.isnan(sar.nrcs_cross.values), 0, sar.nrcs_cross.values)
        x1st = np.argmin(np.abs(sar.lon.values-lon1st))
        y1st = np.argmin(np.abs(sar.lat.values-lat1st))
        res_km = 110 * abs(sar.lat.values[0]-sar.lat.values[1])
        radius = int(rect_radius/res_km)
        if dist_thresh is not None:
            dist_thresh = dist_thresh/res_km

        # 2nd estimation
        nrcs_pad = np.pad(nrcs, radius, mode="empty")
        nrcs_pad = SAR.lee_filter(nrcs_pad, 3)
        x1st, y1st = x1st + radius, y1st + radius
        nrcs_sub = nrcs_pad[y1st-radius:y1st+radius, x1st-radius:x1st+radius]
        tce0 = nrcs_sub < 0.5 * np.nanmean(nrcs_sub)
        lbl, nlbl = ndi.label(tce0)
        centers = ndi.measurements.center_of_mass(tce0, lbl, index=np.arange(1,nlbl+1))

        dist_from_1st = np.sum((np.array(centers) - np.array((nrcs_sub.shape[0]//2, nrcs_sub.shape[1]//2)))**2, axis=1)
        y2nd, x2nd = centers[np.argmin(dist_from_1st)]

        # ith estimation
        polar_radius = int(polar_radius/res_km)
        xi_pad, yi_pad = x1st + x2nd - radius, y1st + y2nd - radius

        break_loop = False
        for p in range(max_loop):
            polar = SAR.get_polar(nrcs_pad, polar_radius, polar_deg, xi_pad, yi_pad)
            nrcs_diff = np.diff(polar.astype(np.float), axis=1)
            nrcs_diff = ndi.gaussian_filter(nrcs_diff, sigma=5)    
            eye_extents = np.argmax(nrcs_diff, axis=1)
            eye_region_polar = np.zeros_like(polar)

            for i, eye_r in enumerate(eye_extents):
                eye_region_polar[i, :eye_r] = 1
            eye_region = SAR.get_cart(eye_region_polar, polar_radius, radius * 2, radius * 2)

            prev_x, prev_y = xi_pad, yi_pad
            ydiff, xdiff = ndi.measurements.center_of_mass(eye_region) - np.array([radius, radius])        
            if dist_thresh is not None and xdiff**2 + ydiff**2 <= dist_thresh ** 2:
                break_loop = True
            xi_pad, yi_pad = xdiff + prev_x, ydiff + prev_y

            plt.imshow(polar, aspect="auto", cmap="gray")
            plt.plot(eye_extents, np.arange(eye_extents.size), c="r")
            plt.show()
            if break_loop:
                break
        xith, yith = xi_pad - radius, yi_pad - radius
        return nrcs_sub, xith, yith

    @staticmethod
    def center_estimation_Combot2020(sar, lon1st, lat1st, hetero_thresh=0.5):
        hetero_mask = sar.heterogeneity_mask.values < 0.5
        xx, yy = np.meshgrid(sar.lon, sar.lat)
        xx -= lon1st
        yy -= lat1st
        dist = np.sqrt(xx**2 + yy**2) * 110 #km
        dist_mask = dist < 50
        co_contrast = np.nanmax(sar.nrcs_co.values[dist_mask]) - np.nanmin(sar.nrcs_co.values[dist_mask])
        cross_contrast = np.nanmax(sar.nrcs_cross.values[dist_mask]) - np.nanmin(sar.nrcs_cross.values[dist_mask])
        co_is_larger_contrast = co_contrast > cross_contrast
        
        nrcs = sar.nrcs_co if co_is_larger_contrast else nrcs_cross
        nrcs_masked = np.where(dist_mask, nrcs, np.nanmax(nrcs))
        lat2nd_ind, lon2nd_ind = np.where(nrcs_masked == np.nanmin(nrcs_masked))
        lon2nd = sar.lon[lon2nd_ind].values.mean()
        lat2nd = sar.lat[lat2nd_ind].values.mean()
        lon2nd_ind = np.where(sar.lon==lon2nd)[0]
        lat2nd_ind = np.where(sar.lat==lat2nd)[0]

        polar_radius = int(100/(110*(sar.lat[0]-sar.lat[1])))
        flags = cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LINEAR
        nrcs_norm = cv2.normalize(nrcs.values * hetero_mask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        nrcs_polar = cv2.warpPolar(nrcs_norm, (polar_radius, int(360/0.5)), (lon2nd_ind, lat2nd_ind), polar_radius, flags) 
        wind_norm = cv2.normalize(sar.wind_speed.values * hetero_mask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        wind_polar = cv2.warpPolar(wind_norm, (polar_radius, int(360/0.5)), (lon2nd_ind, lat2nd_ind), polar_radius, flags) 
        
        nrcs_diff = np.diff(nrcs_polar.astype(np.float), axis=1)
        wind_diff = np.diff(wind_polar.astype(np.float), axis=1)
        diff = nrcs_diff + wind_diff
        blurred_diff = ndi.gaussian_filter(diff, sigma=5)
        
        eye_extents = np.argmax(blurred_diff, axis=1) # align with 0.5 deg
        dirs_crockwise_from_east = np.arange(0, 360, 0.5)

        eye_ext_xs = eye_extents * np.cos(np.deg2rad(dirs_crockwise_from_east)) + lon2nd_ind
        eye_ext_ys = eye_extents * np.sin(np.deg2rad(dirs_crockwise_from_east)) + lat2nd_ind

        x = int(eye_ext_xs.mean())
        y = int(eye_ext_ys.mean())
        lon = sar.lon.values[x]
        lat = sar.lat.values[y]
        misc = {"lon2nd":lon2nd, "lat2nd":lat2nd, "eye_xs":eye_ext_xs, "eye_ys":eye_ext_ys}
        return lon, lat, misc

    @staticmethod
    def center_estimation_Braun2002(sar, lon1st, lat1st, area_size=200, calc_radius=60, threshold=None, debug=False):
        """
        Parameters
        ----------
        sar : xr.Dataset
            SAR data.
        lon1st : int or float
        lat1st : int or float
        area_size : int or float, default 200
            Size (width, height) of the search area (in km).
            Latices inside this area are candidates of center position.
        calc_radius : int or float, default 60
            Calcuration radius of each search point (in km).
        threshold : None or scalar, default None
            The points under the threshold are ignored for center definition.
        debug : bool, default False
            Returns bbox, std2d, cx, cy, area_x_center, area_y_center
        """
        values = sar.wind_speed.values
        mask = sar.heterogeneity_mask.values < 0.5
        mask *= ~np.isnan(values)
        res_km = float(110 * abs(sar.lat[0]-sar.lat[1]))
        # Calcuration area
        calc_radius_px = int(round(calc_radius/res_km))
        values = np.pad(values, calc_radius_px, mode="empty")
        mask = np.pad(mask, calc_radius_px, mode="empty")
        Y, X = np.ogrid[-calc_radius_px:calc_radius_px+1, -calc_radius_px:calc_radius_px+1]
        rlabel = np.round(np.hypot(Y, X)).astype(np.int)
        rlabel[rlabel > calc_radius_px] = 0
        # Search area
        area_x_center = calc_radius_px + np.argmin(np.abs(sar.lon.values-lon1st))
        area_y_center = calc_radius_px + np.argmin(np.abs(sar.lat.values-lat1st))
        area_size_pixel = area_size/res_km
        search_area_x = (area_x_center + np.arange(area_size_pixel) - area_size_pixel/2).astype("int")
        search_area_y = (area_y_center + np.arange(area_size_pixel) - area_size_pixel/2).astype("int")
        search_area_xgrid, search_area_ygrid = np.meshgrid(search_area_x, search_area_y)
        bbox = (search_area_x[0], search_area_x[-1], search_area_y[0], search_area_y[-1])
        # Calcurate
        np_nanmean = np.nanmean
        std_means = []
        for search_x, search_y in zip(search_area_xgrid.flatten(), search_area_ygrid.flatten()):
            calc_x_area = slice(search_x-calc_radius_px, search_x+calc_radius_px+1)
            calc_y_area = slice(search_y-calc_radius_px, search_y+calc_radius_px+1)
            stds = ndi.standard_deviation(values[calc_y_area, calc_x_area], rlabel*mask[calc_y_area, calc_x_area], index=np.arange(calc_radius_px)+1)
            # print(stds)
            # plt.imshow(rlabel*mask[calc_y_area, calc_x_area] == 0)
            # return 0, 0
            std_means.append(np_nanmean(stds))
        std2d = np.array(std_means).reshape(search_area_xgrid.shape)
        # Index of minimum value
        cy, cx = np.unravel_index(np.nanargmin(std2d), std2d.shape)
        plt.imshow(std2d)
        plt.scatter(cx, cy, c="r")
        cx += bbox[0] - calc_radius_px
        cy += bbox[2] - calc_radius_px
        if debug is True:
            return bbox, std2d, cx, cy, area_x_center, area_y_center
        else:
            return sar.lon.values[cx], sar.lat.values[cy]
