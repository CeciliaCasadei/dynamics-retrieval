# -*- coding: utf-8 -*-

import os
import pickle

import matplotlib.pyplot
import numpy
import pylab as pl
import settings
from scipy.io import netcdf

datatype = settings.datatype
results_dir = settings.results_path

longitude_range = [28.0, 295.0]  # EAST
latitude_range = [-60, 20]  # NORTH

# Read NetCDF file from National Center of Atmospheric Research
directory = "../data_CCSM4_Indo_Pacific/raw_data/"
print("Ocean post processed data, monthly averages. Dataset in Giannakis et al.")

listfiles = open("%s/list.txt" % directory, "r")

n_times = 0
for filename in listfiles:
    if ".nc" in filename:

        filename = filename.split()
        fn = filename[-1]

        print(fn)

        f = netcdf.netcdf_file("%s%s" % (directory, fn), "r")

        time_temp = f.variables["time"]
        time = time_temp[:].copy()
        print("time units: ", time_temp.units)
        print("time shape: ", time.shape, "\n")
        n_time_pts = time.shape[0]
        n_times = n_times + n_time_pts

        ulat_temp = f.variables["ULAT"]
        ulat_ref = ulat_temp[:].copy()

        tlat_temp = f.variables["TLAT"]
        tlat_ref = tlat_temp[:].copy()

        ulon_temp = f.variables["ULONG"]
        ulon_ref = ulon_temp[:].copy()

        tlon_temp = f.variables["TLONG"]
        tlon_ref = tlon_temp[:].copy()

listfiles.close()

lon_extremes = tlon_ref[0, :]
idxs = numpy.where(
    (lon_extremes >= longitude_range[0]) & (lon_extremes <= longitude_range[1])
)
idxs_lon = numpy.asarray(idxs)
idxs_lon = idxs_lon.flatten()

lat_extremes = tlat_ref[:, 0]
idxs = numpy.where(
    (lat_extremes >= latitude_range[0]) & (lat_extremes <= latitude_range[1])
)
idxs_lat = numpy.asarray(idxs)
idxs_lat = idxs_lat.flatten()

listfiles = open("%s/list.txt" % directory, "r")

times = numpy.zeros((n_times,))
print("Times shape: ", times.shape)
SST = numpy.zeros((n_times, idxs_lat.shape[0], idxs_lon.shape[0]), dtype=datatype)
print("SST shape: ", SST.shape)

n_times = 0
i = 0
for filename in listfiles:
    if ".nc" in filename:
        i = i + 1

        filename = filename.split()
        fn = filename[-1]

        print("\n")
        print(fn)

        f = netcdf.netcdf_file("%s%s" % (directory, fn), "r")

        time_temp = f.variables["time"]
        time = time_temp[:].copy()
        print("time units: ", time_temp.units)
        print("time shape: ", time.shape)
        start = n_times
        n_time_pts = time.shape[0]
        n_times = n_times + n_time_pts
        end = n_times
        times[start:end] = time

        print("START: ", start)
        print("END: ", end)

        ulat_temp = f.variables["ULAT"]
        ulat = ulat_temp[:].copy()
        if i == 1:
            print("ulat units: ", ulat_temp.units)
            print("ulat shape: ", ulat.shape)
        print("DIFFERENCE TO REF: ", numpy.amax(abs(ulat - ulat_ref)))

        tlat_temp = f.variables["TLAT"]
        tlat = tlat_temp[:].copy()
        if i == 1:
            print("tlat units: ", tlat_temp.units)
            print("tlat shape: ", tlat.shape)
        print("DIFFERENCE TO REF: ", numpy.amax(abs(tlat - tlat_ref)))

        ulon_temp = f.variables["ULONG"]
        ulon = ulon_temp[:].copy()
        if i == 1:
            print("ulon units: ", ulon_temp.units)
            print("ulon shape: ", ulon.shape)
        print("DIFFERENCE TO REF: ", numpy.amax(abs(ulon - ulon_ref)))

        tlon_temp = f.variables["TLONG"]
        tlon = tlon_temp[:].copy()
        if i == 1:
            print("tlon units: ", tlon_temp.units)
            print("tlon shape: ", tlon.shape)
        print("DIFFERENCE TO REF: ", numpy.amax(abs(tlon - tlon_ref)))

        data_temp = f.variables["SST"]
        data = data_temp[:].copy()
        data = data.squeeze()
        print("data units: ", data_temp.units)
        print("data shape: ", data.shape)
        data = data[:, idxs_lat[0] : idxs_lat[-1] + 1, idxs_lon[0] : idxs_lon[-1] + 1]
        print("data shape: ", data.shape, "\n")
        SST[start:end, :, :] = data

listfiles.close()

matplotlib.pyplot.plot(list(range(times.shape[0])), times)
print("\nTIMES IN: ", times[0], times[-1])

# Deal with NaN values
nan_value = numpy.amax(data)
print("NaN value ", nan_value)
print("Replace by NaN in data matrix. \n")
SST[SST == nan_value] = numpy.nan

# Plot raw data
ifPlot = 0

if ifPlot == 1:
    img_dir = "%s/raw_frames" % results_dir
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)

    pl.ion()
    pl.imshow(SST[0, :, :], origin="lower", cmap="jet", extent=[0, 350, 0, 221])
    pl.colorbar()
    for i in range(1200):
        pl.cla()
        pl.imshow(SST[i, :, :], origin="lower", cmap="jet", extent=[0, 350, 0, 221])
        ax = pl.axes()

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        pl.title("%d" % i)
        pl.savefig("%s/%05d.png" % (img_dir, i))
    pl.ioff()
    print("Done!")

# Build masks
print("Build ocean/continent mask (1=ocean, 0=continent)")
data_slice = SST[0, :, :]
print("2D slice shape: ", data_slice.shape)
mask = numpy.ones(data_slice.shape, dtype=numpy.uint8)
mask[numpy.isnan(data_slice)] = 0
print("Mask shape: ", mask.shape, "\n")

# N. grid pts
n_tot_grid_pts = data_slice.shape[0] * data_slice.shape[1]
print("Tot n. of grid points: ", n_tot_grid_pts)
n_ocean_grid_pts = int(mask.sum())
print("Tot n. of ocean grid points: ", n_ocean_grid_pts, "\n")

# Build anomaly matrix
print("Build T anomaly matrix.\n")
SST_ave = numpy.average(SST, axis=0)
print(SST_ave.shape)
SST_anomaly = numpy.ones(SST.shape, dtype=datatype)
for i in range(n_times):
    SST_anomaly[i, :, :] = SST[i, :, :] - SST_ave

print("Anomalies in: ", numpy.nanmin(SST_anomaly), numpy.nanmax(SST_anomaly), "\n")

ifPlot = 0

if ifPlot == 1:
    img_dir = "%s/raw_frames_anomalies" % results_dir
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)

    pl.ion()
    pl.imshow(SST_anomaly[0, :, :], origin="lower", cmap="jet", extent=[0, 350, 0, 221])
    pl.colorbar()
    for i in range(1200):
        pl.cla()
        pl.imshow(
            SST_anomaly[i, :, :], origin="lower", cmap="jet", extent=[0, 350, 0, 221]
        )
        ax = pl.axes()

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        pl.title("%d" % i)
        pl.savefig("%s/%05d.png" % (img_dir, i))
    pl.ioff()
    print("Done!")

# Build 2D data matrix (only ocean pixels)
print("Build 2D data matrix x by flattening each 221x238 matrix to a vector")
print("Use Fortran order in flattening i.e. 238 chunks of 221 elements")
print("Keep only the ocean pixels")
print("Result  x: ", n_ocean_grid_pts, "x", n_times, "\n")

# RESHAPE
# NB:  The rows are made of 238 chunks of 221 elements
SST_reshaped = SST.reshape(n_times, SST.shape[1] * SST.shape[2], order="F")
SST_anomaly_reshaped = SST_anomaly.reshape(
    n_times, SST.shape[1] * SST.shape[2], order="F"
)

# Remove continent pixels
SST_anomaly_ocean = numpy.ones((n_times, n_ocean_grid_pts), dtype=datatype)

n = 0
for i in range(n_tot_grid_pts):
    if numpy.isnan(SST_reshaped[0, i]):
        if not numpy.isnan(SST_anomaly_reshaped[0, i]):
            print("Problem!")
        continue
    else:
        # copy column
        SST_anomaly_ocean[:, n] = SST_anomaly_reshaped[:, i]
        n = n + 1

if n != n_ocean_grid_pts:
    print("Problem!")

x = SST_anomaly_ocean.T
print("Output data anomaly matrix x:", x.shape[0], "x", x.shape[1], "\n")

# Store results
results_dir = settings.results_path
f = open("%s/T_anomaly.pkl" % results_dir, "wb")
pickle.dump(x, f)
f.close()
f = open("%s/mask.pkl" % results_dir, "wb")
pickle.dump(mask, f)
f.close()
