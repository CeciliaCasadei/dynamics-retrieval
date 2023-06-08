# -*- coding: utf-8 -*-
import h5py
import numpy

path = "/sf/alvra/data/p18594/raw"
scan_n = 1
run_n = 1145

dataFile = h5py.File(
    "%s/timetool/rho_nlsa_scan_%d/run_%0.6d.SPECENC.h5" % (path, scan_n, run_n), "r"
)
print(list(dataFile.keys()))

arr_time_SPECENC = dataFile["arrival_times"][:]

arr_time_reliability = dataFile["arrival_times_amplitude"][:]

nom_delay_SPECENC = dataFile["nominal_delay_from_stage"][:]

pulse_ids_SPECENC = dataFile["pulse_ids"][:]

timestamp = arr_time_SPECENC + nom_delay_SPECENC

# arrival time: fs
# arrival time amplitude : tells how reliable arr time is
# nominal delay from stage : scan value  in fs (1 per run)
# pulse id

# timestamp = arrival time [fs] + nominal delay [fs]

dataFile = h5py.File(
    "%s/rho_nlsa_scan_%d/run_%0.6d.BSREAD.h5" % (path, scan_n, run_n), "r"
)
print(list(dataFile.keys()))
data = dataFile["/data/SLAAR11-LMOT-M451:ENC_1_BS/data"]
data = numpy.asarray(data)

time_zero = 186.93624  # 23 July 2020 at 17:30
nom_delay_BSREAD = 2 * (data - time_zero) / 0.0003

# pulse_ids_BSREAD = dataFile['/data/SLAAR11-LMOT-M451:ENC_1_BS/pulse_id']

pulse_Is_BSREAD = dataFile["data/SAROP11-PBPS122:INTENSITY/data"][:]
pulse_ids_BSREAD = dataFile["data/SAROP11-PBPS122:INTENSITY/pulse_id"][:]

diff = abs(pulse_ids_SPECENC - pulse_ids_BSREAD)
diff = diff.sum()

if diff == 0:
    print("OK")

# data = delay line in mm
# nominal delay time = 2(data - t0[mm])/c
# Per-shot correction to noiminal delay
# from script in
# https://jupytera01.psi.ch/user/casadei_c/notebooks/p18594/philip/Jitter_Extractor_karol.ipynb
# correction includes any derift and jitter.

dataFile = h5py.File(
    "%s/rho_nlsa_scan_%d/run_%0.6d.JF06T08V01.h5" % (path, scan_n, run_n), "r"
)
print(list(dataFile.keys()))
print(list(dataFile["general"].keys()))

data = dataFile["data/JF06T08V01/data"]
data = numpy.asarray(data)
# data: 1'000 x 4'112 x 1'030
# 1 frame: (4'112 x 1'030)
# position 1:1000 correspondsto Event n.

pulse_ids_JF = dataFile["data/JF06T08V01/pulse_id"][:].flatten()

diff = abs(pulse_ids_SPECENC - pulse_ids_JF)
diff = diff.sum()

if diff == 0:
    print("OK")
