#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue October 11 09:54:00 2022
    Copyright (C) <2022>  <Håvard Boutera Toft>
    htla@nve.no

    This python script reimplements the Potential Release Area proposed 
    by Veitinger et al. (2016) and Sharp et al., (2018). The script has
    been modified to suit AutoATES v2.0 and is rewritten using Python
    libraries.

    References:
    https://github.com/jocha81/Avalanche-release
        Veitinger, J., Purves, R. S., & Sovilla, B. (2016). Potential 
    slab avalanche release area identification from estimated winter 
    terrain: a multi-scale, fuzzy logic approach. Natural Hazards and 
    Earth System Sciences, 16(10), 2211-2225.
        Sharp, A. E. A. (2018). Evaluating the Exposure of Heliskiing 
    Ski Guides to Avalanche Terrain Using a Fuzzy Logic Avalanche 
    Susceptibility Model. University of Leeds: Leeds, UK.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

    Description of inputs and defaults.
        forest_type:    'stems', 'bav', 'pcc' and 'no_forest'
        DEM:            A raster using the GeoTiff format (int16, nodata=-9999)
        FOREST:         A raster using the GeoTiff format (int16, nodata=0)
        radius:         The radius of the windshelter function. A general recommendation is to use 60 m, so if the cell size is 10 m, the radius should be 6.
        prob:           Default is 0.5, (see Veitinger et al. 2016 for more information).
        winddir:        The prevailing wind direction (0-360). Default for AutoATES v2.0 is 0
        windtol:        The number of degrees to each side of the prevailing wind (0-180). Default for AutoATES v2.0 is 180.
        pra_thd:        The cut off value for the binary PRA output. Default for AutoATES is 0.15
        sf:             The SieveFilter removes small clusters of cells smaller than the dessignated value in the binary PRA output. I.e., sf=3 means that release areas with less than 3 cells will be made no release cell.
"""

# import standard libraries
import numpy as np
import rasterio, rasterio.mask
from osgeo import gdal
import os
from numpy.lib.stride_tricks import as_strided
from collections import deque
import sys
from datetime import datetime

# --- Example
# stems (10m raster):       python PRA/PRA_AutoATES-v2.0.py stems PRA/DEM.tif PRA/FOREST.tif 6 0.5 0 180 0.15 3
# no_forest (10m raster):   python PRA/PRA_AutoATES-v2.0.py no_forest PRA/DEM.tif 6 0.5 0 180 0.15 3

def PRA(forest_type, DEM, FOREST, radius, prob, winddir, windtol, pra_thd, sf):
    
    ##########################
    # --- Check input files
    ##########################

    path = os.path.join(os.getcwd(), "PRA")
    os.makedirs(path, exist_ok=True)

    f= open("PRA/log.txt","w+")
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    f.write("Start time = {}\n".format(current_time))

    # Check if path exits
    if os.path.exists(DEM) is False:
        print("The DEM path {} does not exist".format(DEM))

    if forest_type in ['pcc', 'stems', 'bav']:
        # Check if path exits
        if os.path.exists(FOREST) is False:
            print("The forest path {} does not exist\n".format(FOREST))

        print(forest_type, DEM, FOREST, radius, prob, winddir, windtol, pra_thd, sf)
        f.write("forest_type: {}, DEM: {}, FOREST: {}, radius: {}, prob: {}, winddir: {}, windtol {}, pra_thd: {}, sf: {}\n".format(forest_type, DEM, FOREST, radius, prob, winddir, windtol, pra_thd, sf))

    if forest_type in ['no_forest']:
        print(forest_type, DEM, radius, prob, winddir, windtol, pra_thd, sf)
        f.write("forest_type: {}, DEM: {}, radius: {}, prob: {}, winddir: {}, windtol {}, pra_thd: {}, sf: {}\n".format(forest_type, DEM, radius, prob, winddir, windtol, pra_thd, sf))

    #########################
    # --- Define functions
    #########################

    def sliding_window_view(arr, window_shape, steps):
        """ 
        Produce a view from a sliding, striding window over `arr`.
        The window is only placed in 'valid' positions - no overlapping
        over the boundary
        """

        in_shape = np.array(arr.shape[-len(steps):])  # [x, (...), z]
        window_shape = np.array(window_shape)  # [Wx, (...), Wz]
        steps = np.array(steps)  # [Sx, (...), Sz]
        nbytes = arr.strides[-1]  # size (bytes) of an element in `arr`

        # number of per-byte steps to take to fill window
        window_strides = tuple(np.cumprod(arr.shape[:0:-1])[::-1]) + (1,)
        # number of per-byte steps to take to place window
        step_strides = tuple(window_strides[-len(steps):] * steps)
        # number of bytes to step to populate sliding window view
        strides = tuple(int(i) * nbytes for i in step_strides + window_strides)

        outshape = tuple((in_shape - window_shape) // steps + 1)
        # outshape: ([X, (...), Z], ..., [Wx, (...), Wz])
        outshape = outshape + arr.shape[:-len(steps)] + tuple(window_shape)
        return as_strided(arr, shape=outshape, strides=strides, writeable=False)

    def sector_mask(shape,centre,radius,angle_range): # used in windshelter_prep
        """
        Return a boolean mask for a circular sector. The start/stop angles in  
        `angle_range` should be given in clockwise order.
        """

        x,y = np.ogrid[:shape[0],:shape[1]]
        cx,cy = centre
        tmin,tmax = np.deg2rad(angle_range)

        # ensure stop angle > start angle
        if tmax < tmin:
                tmax += 2*np.pi

        # convert cartesian --> polar coordinates
        r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
        theta = np.arctan2(x-cx,y-cy) - tmin

        # wrap angles between 0 and 2*pi
        theta %= (2*np.pi)

        # circular mask
        circmask = r2 <= radius*radius

        # angular mask
        anglemask = theta <= (tmax-tmin)

        a = circmask*anglemask

        return a

    def windshelter_prep(radius, direction, tolerance, cellsize):
        x_size = y_size = 2*radius+1
        x_arr, y_arr = np.mgrid[0:x_size, 0:y_size]
        cell_center = (radius, radius)
        dist = (np.sqrt((x_arr - cell_center[0])**2 + (y_arr - cell_center[1])**2))*cellsize
        # dist = np.round(dist, 5)

        mask = sector_mask(dist.shape, (radius, radius), radius, (direction, tolerance))
        mask[radius, radius] = True # bug fix

        return dist, mask

    def windshelter(x, prob, dist, mask, radius): # applying the windshelter function
        data = x*mask
        data[data==profile['nodata']]=np.nan
        data[data==0]=np.nan
        center = data[radius, radius]
        data[radius, radius]=np.nan
        data = np.arctan((data-center)/dist)
        data = np.nanquantile(data, prob)
        return data

    def windshelter_window(radius, prob):

        dist, mask = windshelter_prep(radius, winddir - windtol + 270, winddir + windtol + 270, cell_size)
        window = sliding_window_view(array[-1], ((radius*2)+1,(radius*2)+1), (1, 1))

        nc = window.shape[0]
        nr = window.shape[1]
        ws = deque()

        for i in range(nc):
            for j in range(nr):
                data = window[i, j]
                data = windshelter(data, prob, dist, mask, radius).tolist()
                ws.append(data)

        data = np.array(ws)
        data = data.reshape(nc, nr)
        data = np.pad(data, pad_width=radius, mode='constant', constant_values=-9999)
        data = data.reshape(1, data.shape[0], data.shape[1])
        data = data.astype('float32')
        
        return data

    #######################
    # Calculate slope and windshelter
    #######################
    
    print("Calculating slope angle")

    with rasterio.open(DEM) as src:
        array = src.read(1)
        profile = src.profile
        array[np.where(array < -100)] = -9999
        
    cell_size = profile['transform'][0]

    # Evaluate gradient in two dimensions
    px, py = np.gradient(array, cell_size)
    slope = np.sqrt(px ** 2 + py ** 2)

    # If needed in degrees, convert using
    slope_deg = np.degrees(np.arctan(slope))

    print("Calculating windshelter")

    # Calculate windshelter
    with rasterio.open(DEM) as src:
        array = src.read()
        array = array.astype('float')
        profile = src.profile
        cell_size = profile['transform'][0]
    data = windshelter_window(radius, prob)

    with rasterio.open(DEM) as src:
        profile = src.profile
    profile.update({"dtype": "float32", "nodata": -9999})

    data = np.nan_to_num(data, nan=-9999)

    # Save raster to path using meta data from dem.tif (i.e. projection)
    with rasterio.open('PRA/windshelter.tif', "w", **profile) as dest:
        dest.write(data)

    print("Defining Cauchy functions")

    #######################
    # --- Cauchy functions
    #######################

    # --- Define bell curve parameters for slope
    a = 11
    b = 4
    c = 43

    f.write("Cauchy slope function: a={}, b={}, c={}\n".format(a, b, c))

    slopeC = 1/(1+((slope_deg-c)/a)**(2*b))

    # --- Define bell curve parameters for windshelter
    a = 3
    b = 10
    c = 3
    f.write("Cauchy windshelter function: a={}, b={}, c={}\n".format(a, b, c))

    with rasterio.open("PRA/windshelter.tif") as src:
        windshelter = src.read()

    windshelterC = 1/(1+((windshelter-c)/a)**(2*b))

    # --- Define bell curve parameters for forest stem density
    if forest_type in ['stems']:
        a = 350
        b = 2
        c = -120
        f.write("Cauchy forest function (stems): a={}, b={}, c={}\n".format(a, b, c))

    if forest_type in ['bav']:
        a = 20
        b = 3.5
        c = -10
        f.write("Cauchy forest function (bav): a={}, b={}, c={}\n".format(a, b, c))

    # --- Define bell curve parameters for percent canopy cover
    if forest_type in ['pcc', 'no_forest']:
        a = 40
        b = 3.5
        c = -15

        if forest_type in ['pcc']:
            f.write("Cauchy forest function (pcc): a={}, b={}, c={}\n".format(a, b, c))
        if forest_type in ['no_forest']:
            f.write("No forest input given\n")

    if forest_type in ['pcc', 'stems']:
        with rasterio.open(FOREST) as src:
            forest = src.read()

    if forest_type in ['no_forest']:
        with rasterio.open(DEM) as src:
            forest = src.read()
            forest = np.where(forest > -100, 0, forest)

    forestC = 1/(1+((forest-c)/a)**(2*b))
    # --- Ares with no forest and assigned -9999 will get a really small value which suggest dense forest. This function fixes this, but might have to be adjusted depending on the input dataset.
    forestC[np.where(forestC <= 0.00001)] = 1

    slopeC = np.round(slopeC, 5)
    windshelterC = np.round(windshelterC, 5)
    forestC = np.round(forestC, 5)

    #######################
    # --- Fuzzy logic operator
    #######################

    print("Starting the Fuzzy Logic Operator")

    minvar = np.minimum(slopeC, windshelterC)
    minvar = np.minimum(minvar, forestC)

    PRA = (1-minvar)*minvar+minvar*(slopeC+windshelterC+forestC)/3
    PRA = np.round(PRA, 5)
    PRA = PRA * 100

    # --- Update metadata
    profile.update({'dtype': 'int16', 'nodata': -9999})

    # --- Save raster to path using meta data from dem.tif (i.e. projection)
    with rasterio.open('PRA/PRA_continous.tif', "w", **profile) as dest:
        dest.write(PRA.astype('int16'))

    # --- Reclassify PRA to be used as input for FlowPy
    profile.update({'nodata': -9999})
    pra_thd = pra_thd * 100
    PRA[np.where((0 <= PRA) & (PRA < pra_thd))] = 0
    PRA[np.where((pra_thd < PRA) & (PRA <= 100))] = 1

    with rasterio.open('PRA/PRA_binary.tif', "w", **profile) as dest:
        dest.write(PRA.astype('int16'))

    # --- Remove islands smaller than 3 pixels
    sievefilter = sf + 1
    Image = gdal.Open('PRA/PRA_binary.tif', 1)  # open image in read-write mode
    Band = Image.GetRasterBand(1)
    gdal.SieveFilter(srcBand=Band, maskBand=None, dstBand=Band, threshold=sievefilter, connectedness=8, callback=gdal.TermProgress_nocb)
    del Image, Band  # close the datasets.
    
    print('PRA complete')
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    f.write("Stop time = {}\n".format(current_time))
    f.close()

if __name__ == "__main__":
    forest_type = str(sys.argv[1])
    if forest_type in ['pcc', 'stems', 'bav']:
        DEM = sys.argv[2]
        FOREST = sys.argv[3]
        radius = int(sys.argv[4])
        prob = float(sys.argv[5])
        winddir = int(sys.argv[6])
        windtol = int(sys.argv[7])
        pra_thd = float(sys.argv[8])
        sf = int(sys.argv[9])
        PRA(forest_type, DEM, FOREST, radius, prob, winddir, windtol, pra_thd, sf)
    if forest_type in ['no_forest']:
        DEM = sys.argv[2]
        radius = int(sys.argv[3])
        prob = float(sys.argv[4])
        winddir = int(sys.argv[5])
        windtol = int(sys.argv[6])
        pra_thd = float(sys.argv[7])
        sf = int(sys.argv[8])
        PRA(forest_type, DEM, DEM, radius, prob, winddir, windtol, pra_thd, sf)
