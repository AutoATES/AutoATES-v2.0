import numpy as np
import rasterio, rasterio.mask
from osgeo import gdal
import os
from skimage import morphology
import csv
import scipy.ndimage
from rasterio.fill import fillnodata

# --- Set Input Files
DEM = 'test-data/Bow Summit/dem.tif'
canopy = 'test-data/Bow Summit/forest.tif'
cell_count = 'test-data/Bow Summit/Overhead.tif' # replace with z_delta in next iteration
FP = 'test-data/Bow Summit/FP_int16.tif'
SZ = 'test-data/Bow Summit/pra_binary.tif'
forest_type = 'bav' # 'bav', 'stems', 'pcc', 'sen2cc'

wd = 'test-data/Bow Summit/outputs'

# --- Set default input parameters

# Moving window size to smooth slope angle layer for calcuation of Class 4 extreme
WIN_SIZE= 3

# --- Define slope angle Thresholds
# Should I increase these to capture more real world numbers or keep values based on Consensus map test areas?
# Class 0 / 1 Slope Angle Threshold (Default 15)
SAT01 = 15
# Class 1 / 2 Slope Angle Threshold (Default 18)
SAT12 = 18
# Class 2 / 3 Slope Angle Threshold (Default 28)
SAT23 = 28
# Class 3 / 4 Slope Angle Threshold (Default 39)
# This is calculated on a smoothed raster layer, so the slope angle value is not representative of real world values
SAT34 = 39 # stereo

# --- Define alpha angle thresholds
# Class 1 Alpha Angle Threshold (Default 18)
AAT1 = 18
# Class 2 Alpha Angle Threshold (Default 25)
AAT2 = 24
# Class 3 Alpha Angle Threshold (Default 38)
AAT3 = 33

if forest_type in ['pcc']:
    # --- Add tree coverage criteria
    # Tree classification: "open" (upper bound)
    TREE1 = 10
    # Tree classification: "sparse" (upper bound)
    TREE2 = 50
    # Tree classification: "mixed" (upper bound)
    TREE3 = 65

if forest_type in ['bav']:
    # --- Add tree coverage criteria
    # Tree classification: "open" (upper bound)
    TREE1 = 10
    # Tree classification: "sparse" (upper bound)
    TREE2 = 20
    # Tree classification: "mixed" (upper bound)
    TREE3 = 25

if forest_type in ['stems']:
    # --- Add tree coverage criteria
    # Tree classification: "open" (upper bound)
    TREE1 = 100
    # Tree classification: "sparse" (upper bound)
    TREE2 = 250
    # Tree classification: "mixed" (upper bound)
    TREE3 = 500

if forest_type in ['sen2ccc']:
    # --- Add tree coverage criteria
    # Tree classification: "open" (upper bound)
    TREE1 = 100 # dummy values
    # Tree classification: "sparse" (upper bound)
    TREE2 = 250 # dummy values
    # Tree classification: "mixed" (upper bound)
    TREE3 = 500 # dummy values

# --- Add cell count criteria
CC1 = 5
CC2 = 40

# --- Threshold for number of cells in a cluster to be removed (generalization)
ISL_SIZE = 30000

def AutoATES(wd, DEM, canopy, cell_count, FP, SAT01, SAT12, SAT23, SAT34, AAT1, AAT2, AAT3, TREE1, TREE2, TREE3, CC1, CC2, ISL_SIZE, WIN_SIZE):
    
    # --- Write input parameters to CSV file
    labels = ['DEM', 'canopy', 'cell_count', 'FP', 'SAT01', 'SAT12', 'SAT23', 'SAT34', 'AAT1', 'AAT2', 'AAT3', 'TREE1', 'TREE2', 'TREE3', 'CC1', 'CC2', 'ISL_SIZE', 'WIN_SIZE']
    csvRow = [DEM, canopy, cell_count, FP, SAT01, SAT12, SAT23, SAT34, AAT1, AAT2, AAT3, TREE1, TREE2, TREE3, CC1, CC2, ISL_SIZE, WIN_SIZE]
    csvfile = os.path.join(wd, "inputpara.csv")
    with open(csvfile, "a") as fp:
        wr = csv.writer(fp, dialect='excel')
        wr.writerow(labels)
        wr.writerow(csvRow)
    
    # --- Calculate slope angle
    def calculate_slope(DEM):
        gdal.DEMProcessing(os.path.join(wd, 'slope.tif'), DEM, 'slope')
        with rasterio.open(os.path.join(wd, 'slope.tif')) as src:
            slope = src.read()
            profile = src.profile
        return slope, profile

    slope, profile = calculate_slope(DEM)
    slope = slope.astype('int16')
    
    slope_nd = np.where(slope < 0, 0, slope)
    
    # Optional function to calculat class 4 slope using a neighborhood function - controlled by WIN_SIZE input parameter
    # If WIN_SIZE is set to 1 this function does not do anything to the SAT34 threshold calculation
    slope_smooth = scipy.ndimage.uniform_filter(slope_nd, size = WIN_SIZE, mode = 'nearest')
    
    # Update metadata
    profile.update({"driver": "GTiff", "nodata": -9999, 'dtype': 'int16'})
    
    # Reclassify
    slope[np.where((0 < slope) & (slope <= SAT01))] = 0
    slope[np.where((SAT01 < slope) & (slope <= SAT12))] = 1
    slope[np.where((SAT12 < slope) & (slope <= SAT23))] = 2
    slope[np.where((SAT23 < slope) & (slope <= 100))] = 3
    slope[np.where((SAT34 < slope_smooth) & (slope_smooth <= 100))] = 4

    with rasterio.open(os.path.join(wd, "slope.tif"), 'w', **profile) as dst:
        dst.write(slope)
        
    with rasterio.open(os.path.join(wd, "slope_smooth.tif"), 'w', **profile) as dst:
        dst.write(slope_smooth)

    # --- Open Flow-Py data, reclassify by thresholds and combine class 1, 2, and 3 runout zones into one raster

    # --- AAT1
    with rasterio.open(FP) as src:
        array = src.read(1)
        profile = src.profile
        array = array.astype('int16')  

    flow_py18 = array
    flow_py18[np.where((flow_py18 >= 0) & (flow_py18 < 90))] = 1 # Changed to 0 from AAT1 because we are not using Non-Avalanche Terrain - class 0

    # --- AAT2
    with rasterio.open(FP) as src:
        array = src.read(1)
        profile = src.profile
        array = array.astype('int16')  

    flow_py25 = array    
    flow_py25[np.where((flow_py25 < AAT2))] = 0
    flow_py25[np.where((flow_py25 >= AAT2) & (flow_py25 < 90))] = 2

    # --- AAT3
    with rasterio.open(FP) as src:
        array = src.read(1)
        profile = src.profile
        array = array.astype('int16')   

    flow_py38 = array
    flow_py38[np.where((flow_py38 < AAT3))] = 0
    flow_py38[np.where((flow_py38 >= AAT3) & (flow_py38 < 90))] = 3

    flowpy = np.maximum(flow_py18, flow_py25)
    flowpy = np.maximum(flowpy, flow_py38)
    flowpy = flowpy.reshape(1, flowpy.shape[0], flowpy.shape[1])

    # Update metadata
    profile.update({"driver": "GTiff", "nodata": -9999, 'dtype': 'int16'})

    with rasterio.open(os.path.join(wd, "flowpy.tif"), 'w', **profile) as dst:
        dst.write(flowpy)

    # --- Add cell count criteria

    # --- Reclassify cell count criteria
    with rasterio.open(cell_count) as src:
        array = src.read()
        array = array.astype('int16')
        profile = src.profile

        # Update metadata
        profile.update({"driver": "GTiff", "nodata": -9999, 'dtype': 'int16'})

        # Reclassify
        array[np.where(array == -9999)] = 0
        array[np.where((0 <= array) & (array <= CC1))] = 1
        array[np.where((CC1 < array) & (array <= CC2))] = 2
        array[np.where((CC2 < array) & (array <= 20000))] = 3

    with rasterio.open(os.path.join(wd, "cellcount_reclass.tif"), 'w', **profile) as dst:
        dst.write(array)

    # --- Combine Tree coverage, slope class and cell count

    src1 = rasterio.open(os.path.join(wd, "slope.tif"))
    src1 = src1.read()

    src2 = rasterio.open(os.path.join(wd, "flowpy.tif"))
    src2 = src2.read()

    src3 = rasterio.open(os.path.join(wd, "cellcount_reclass.tif"))
    src3 = src3.read()

    ates = np.maximum(src1, src2)
    ates = np.maximum(ates, src3)

    with rasterio.open(os.path.join(wd, "merge_new.tif"), 'w', **profile) as dst:
        dst.write(ates)

    # --- Add tree coverage criteria

    src1 = rasterio.open(os.path.join(wd, "merge_new.tif"))
    src1 = src1.read()
    profile.update({"driver": "GTiff", "nodata": -9999, 'dtype': 'int16'})


    # --- Reclassify using the forest criteria
    forest = rasterio.open(canopy).read()
    forest_open=forest
    forest_open[forest_open > TREE1] = -1
    forest_open[(forest_open >= 0) & (forest_open <= TREE1)] = 10
        
    forest = rasterio.open(canopy).read()
    forest_sparse=forest
    forest_sparse[forest_sparse > TREE2] = -1
    forest_sparse[forest <= TREE1] = -1
    forest_sparse[(forest > TREE1) & (forest <= TREE2)] = 20
    
    forest = rasterio.open(canopy).read()
    forest_dense=forest
    forest_dense[forest_dense > TREE3] = -1
    forest_dense[forest_dense <= TREE2] = -1
    forest_dense[(forest_dense > TREE2) & (forest_dense <= TREE3)] = 30
    
    forest = rasterio.open(canopy).read()
    forest_vdense=forest
    forest_vdense[forest_vdense < TREE3] = -1
    forest_vdense[forest_vdense >= TREE3] = 40
    
    src2=np.maximum(forest_open, forest_sparse)
    src2=np.maximum(src2, forest_dense)
    src2=np.maximum(src2, forest_vdense)
    
    with rasterio.open(os.path.join(wd, "forest_reclass.tif"), 'w', **profile) as dst:
        dst.write(src2)
    
    # --- Add PRA criteria
    src3 = rasterio.open(SZ)
    src3 = src3.read()
    
    src3[np.where(0 == src3)] = 0
    src3[np.where(1 == src3)] = 100

    with rasterio.open(os.path.join(wd, "SZ_reclass.tif"), 'w', **profile) as dst:
        dst.write(src3)

    array = np.sum([src1, src2, src3], axis=0)

    array[np.where(array == 10)] = 0
    array[np.where(array == 11)] = 1
    array[np.where(array == 12)] = 2
    array[np.where(array == 13)] = 3
    array[np.where(array == 14)] = 4
    array[np.where(array == 20)] = 0
    array[np.where(array == 21)] = 1
    array[np.where(array == 22)] = 1
    array[np.where(array == 23)] = 2
    array[np.where(array == 24)] = 3
    array[np.where(array == 30)] = 0
    array[np.where(array == 31)] = 1
    array[np.where(array == 32)] = 1
    array[np.where(array == 33)] = 1
    array[np.where(array == 34)] = 3
    array[np.where(array == 40)] = 0
    array[np.where(array == 41)] = 1
    array[np.where(array == 42)] = 1
    array[np.where(array == 43)] = 1
    array[np.where(array == 44)] = 2
    array[np.where(array == 110)] = 0
    array[np.where(array == 111)] = 1
    array[np.where(array == 112)] = 2
    array[np.where(array == 113)] = 3
    array[np.where(array == 114)] = 4
    array[np.where(array == 120)] = 0
    array[np.where(array == 121)] = 1
    array[np.where(array == 122)] = 1
    array[np.where(array == 123)] = 2
    array[np.where(array == 124)] = 3
    array[np.where(array == 130)] = 0
    array[np.where(array == 131)] = 1
    array[np.where(array == 132)] = 1
    array[np.where(array == 133)] = 2
    array[np.where(array == 134)] = 3
    array[np.where(array == 140)] = 0
    array[np.where(array == 141)] = 1
    array[np.where(array == 142)] = 1
    array[np.where(array == 143)] = 2
    array[np.where(array == 144)] = 2
    array[np.where(array < 0)] = 0

    array = array.astype('int16')

    # --- Save raster to path
    with rasterio.open(os.path.join(wd, "merge_all.tif"), "w", **profile) as dest:
        dest.write(array)

    # --- Remove clusters of raster cells smaller than ISL_SIZE
    raster = gdal.Open(DEM)
    gt =raster.GetGeoTransform()
    pixelSizeX = gt[1]
    pixelSizeY =-gt[5]
    num_cells = np.around(ISL_SIZE / (pixelSizeX * pixelSizeY)) 
    #print(num_cells)
    # --- Open file
    src1 = rasterio.open(os.path.join(wd, "merge_all.tif"))
    src1 = src1.read(1)

    # --- Change values to prepare for morphology and rasterio.fill
    src1 = src1 + 1
    src1 = src1.reshape(1, src1.shape[0], src1.shape[1])

    # --- Same as region group in arcmap. Each cluster gets a value between 1 and num_labels (number of clusters)
    # 20210430 JS changed connectivity to 2
    lab, num_labels = morphology.label(src1, connectivity=2, return_num=True)

    rg = np.arange(1, num_labels+1, 1)

    # --- Loop through all clusters and assign all clusters with less then ISL_SIZE to the value 0 (set null)
    for i in rg:
        occurrences = np.count_nonzero(lab == i)
        if occurrences < num_cells:
            lab[np.where(lab == i)] = 0

    # --- Save as dtype int16
    lab = lab.astype('int16')

    search_dist = num_cells / 4
    #search_dist = num_cells

    # --- This algorithm will interpolate values for all designated nodata pixels (marked by zeros) (nibble)
    data = rasterio.fill.fillnodata(src1, lab, max_search_distance=search_dist, smoothing_iterations=0)
    
    # --- Change values back to standardized way of plotting ATES (0, 1, 2, 3 and 4)
    data = data - 1
    data[np.where(data == 0)] = -9999
    data = data.astype('int16')
    profile.update({"driver": "GTiff", "nodata": -9999, 'dtype': 'int16'})

    # --- Save raster to path
    with rasterio.open(os.path.join(wd, "ates_gen.tif"), "w", **profile) as dest:
        dest.write(data)

if __name__ == "__main__":
    AutoATES(wd, DEM, canopy, cell_count, FP, SAT01, SAT12, SAT23, SAT34, AAT1, AAT2, AAT3, TREE1, TREE2, TREE3, CC1, CC2, ISL_SIZE, WIN_SIZE)