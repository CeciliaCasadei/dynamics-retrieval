# -*- coding: utf-8 -*-
import h5py

myfile = h5py.File(
    "/das/work/p16/p16818/data/scratch/cheetah/hdf5/r0058-br-tt/cxilp4115-r0058-class1-c00.cxi",
    "r",
)
print(list(myfile.keys()))
LCLS = myfile["LCLS"]
print(list(LCLS.keys()))
detector_1 = LCLS["detector_1"]
print(list(detector_1.keys()))
EncoderValue = detector_1["EncoderValue"][:]  # mm
# Encoder value [mm] + coffset [m] = detector distance (average camera length)
