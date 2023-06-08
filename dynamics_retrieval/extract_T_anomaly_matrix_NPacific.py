# -*- coding: utf-8 -*-

import os
import pickle

import matplotlib.pyplot
import numpy
import pylab as pl
import settings_NP as settings
from scipy.io import loadmat, netcdf

print("\n****** RUNNING extract_T_anomaly_matrix_NPacific ******")

# Read NetCDF file from National Center of Atmospheric Research
directory = "../data_Giannakis_PNAS_2012/data_raw/"
fn = "b30.004.T0-300m.0300-01_cat_0999-12.nc"
print("Dataset in Giannakis et al., PNAS, 2012")
print(fn, "\n")

f = netcdf.netcdf_file("%s%s" % (directory, fn), "r")

# Extract data
print("VARIABLES:\n")
print(f.variables, "\n")

data_temp = f.variables["dat"]
data = data_temp[:].copy()
print("Monthly values of sea average T on top 300 m ")
print("data units: ", data_temp.units)
print("data shape: ", data.shape, "\n")

time_temp = f.variables["time"]
time = time_temp[:].copy()
print("time units: ", time_temp.units)
print("time shape: ", time.shape, "\n")
n_time_pts = time.shape[0]

lat_temp = f.variables["lat"]
lat = lat_temp[:].copy()
print("lat units: ", lat_temp.units)
print("lat shape: ", lat.shape)
print("lat in [", lat[0], ", ", lat[-1], "]\n")
n_lat_pts = lat.shape[0]

lon_temp = f.variables["lon"]
lon = lon_temp[:].copy()
print("lon units: ", lon_temp.units)
print("lon shape: ", lon.shape)
print("lon in [", lon[0], ", ", lon[-1], "]\n")
n_lon_pts = lon.shape[0]

f.close()

# Deal with NaN values
nan_value = numpy.amax(data)
print("NaN value ", nan_value)
print("Replace by NaN in data matrix. \n")
data[data == nan_value] = numpy.nan

# Build masks
print("Build ocean/continent mask (1=ocean, 0=continent)")
data_slice = data[0, :, :]
print("2D slice shape: ", data_slice.shape)
mask = numpy.ones(data_slice.shape, dtype=numpy.uint8)
mask[numpy.isnan(data_slice)] = 0
print("Mask shape: ", mask.shape, "\n")

mask_lon = numpy.asarray(
    [
        lon,
    ]
    * n_lat_pts
)
mask_lat = numpy.asarray(
    [
        lat,
    ]
    * n_lon_pts
).T
print("Build latitude and longitude grids: ", mask_lon.shape[0], "x", mask_lon.shape[
    1
], "\n")

# Check mask
ifOcean_dict = loadmat("../data_Giannakis_PNAS_2012/matlab/ifOcean.mat")
ifOcean = ifOcean_dict["ifOcean"]
diff = ifOcean - mask
print("Diff between python and matlab masks: ", numpy.amax(diff), numpy.amin(diff), "\n")

# N. grid pts
n_tot_grid_pts = data_slice.shape[0] * data_slice.shape[1]
print("Tot n. of grid points: ", n_tot_grid_pts)
n_ocean_grid_pts = int(mask.sum())
print("Tot n. of ocean grid points: ", n_ocean_grid_pts, "\n")

# Build anomaly matrix
print("Build T anomaly matrix.\n")
data = numpy.asarray(data, dtype=numpy.float64)
data_ave = numpy.average(data, axis=0)
data_anomaly = numpy.ones(data.shape, dtype=numpy.float64)
for i in range(n_time_pts):
    data_anomaly[i, :, :] = data[i, :, :] - data_ave

print("Anomalies in: ", numpy.nanmin(data_anomaly), numpy.nanmax(data_anomaly), "\n")

# Build 2D data matrix (only ocean pixels)
print("Build 2D data matrix x by flattening each 16x46 matrix to a vector")
print("Use Fortran order in flattening i.e. 46 chunks of 16 elements")
print("Keep only the ocean pixels")
print("Result  x: ", n_ocean_grid_pts, "x", n_time_pts, "\n")

# RESHAPE
# NB:  The rows are made of 46 chunks of 16 elements
data_reshaped = data.reshape(n_time_pts, data.shape[1] * data.shape[2], order="F")
data_anomaly_reshaped = data_anomaly.reshape(
    n_time_pts, data.shape[1] * data.shape[2], order="F"
)

# Remove continent pixels
data_ocean = numpy.ones((n_time_pts, n_ocean_grid_pts))
data_anomaly_ocean = numpy.ones((n_time_pts, n_ocean_grid_pts))

n = 0
for i in range(n_tot_grid_pts):
    if numpy.isnan(data_reshaped[0, i]):
        if not numpy.isnan(data_anomaly_reshaped[0, i]):
            print("Problem!")
        continue
    else:
        # copy column
        data_ocean[:, n] = data_reshaped[:, i]
        data_anomaly_ocean[:, n] = data_anomaly_reshaped[:, i]
        n = n + 1

if n != n_ocean_grid_pts:
    print("Problem!")

print("Output data anomaly matrix x:", n_ocean_grid_pts, "x", n_time_pts, "\n")
x = data_anomaly_ocean.T

data_ocean_T = data_ocean.T

# Compare result
result_dict = loadmat("../data_Giannakis_PNAS_2012/matlab/dataAnomaly.mat")
result_matlab = result_dict["myData"]
diff = x - result_matlab
print("Diff between python and matlab final result x: %.10f %.10f\n" % (
    numpy.amax(diff),
    numpy.amin(diff),
))

# Store results
results_dir = settings.results_path
f = open("%s/T_anomaly.pkl" % results_dir, "wb")
pickle.dump(x, f)
f.close()
f = open("%s/mask.pkl" % results_dir, "wb")
pickle.dump(mask, f)
f.close()
f = open("%s/mask_lat.pkl" % results_dir, "wb")
pickle.dump(mask_lat, f)
f.close()
f = open("%s/mask_lon.pkl" % results_dir, "wb")
pickle.dump(mask_lon, f)
f.close()
f = open("%s/T.pkl" % results_dir, "wb")
pickle.dump(data_ocean_T, f)
f.close()
f = open("%s/T_average.pkl" % results_dir, "wb")
pickle.dump(data_ave, f)
f.close()

print(numpy.nanmin(data), numpy.nanmax(data))
# Plot raw data
ifPlot = 0

if ifPlot == 1:
    img_dir = "%s/raw_frames" % results_dir
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)

    pl.ion()
    # pl.imshow(data_anomaly[0,:-1,:-1], origin='lower', vmin=-1.2, vmax=+1.2, cmap='jet')
    pl.imshow(data[0, :-1, :-1], origin="lower", vmin=-1.0, vmax=+26, cmap="jet")
    pl.colorbar()
    for i in range(1200, 2400):
        pl.cla()
        # pl.imshow(data_anomaly[i,:-1,:-1], origin='lower', vmin=-1.2, vmax=+1.2, cmap='jet')
        pl.imshow(data[i, :-1, :-1], origin="lower", vmin=-1.0, vmax=+26, cmap="jet")
        pl.title("%d" % i)
        pl.savefig("%s/%05d.png" % (img_dir, i))
    pl.ioff()
    print ("Done!")


## Look for 1st frame in PNAS movies
##idxs_candidates = []
##for i in range(n_time_pts):
##    img = data_anomaly[i,:,:]
##    a = img[5:9,9:14]
##    b = img[1:2,39:41]
##    if numpy.all([a>0.95]) and numpy.all([b<-0.75]):
##        idxs_candidates.append(i)
##        matplotlib.pyplot.figure()
##        matplotlib.pyplot.imshow(img, origin='lower', vmin=-1, vmax=+1.2, cmap='jet')
##        matplotlib.pyplot.savefig('./start_candidates/%05d.png'%i)
##        matplotlib.pyplot.close()
