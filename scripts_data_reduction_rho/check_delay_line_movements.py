# -*- coding: utf-8 -*-
import h5py
import matplotlib.pyplot
path = "/das/work/p17/p17491/Cecilia_Casadei/NLSA/data_rho/retrieved_specenc/sf/alvra/data/p18594/raw/timetool/BSREAD"

scan_ns = range(1, 24)
positions_all = []
for scan_n in scan_ns:
    print scan_n
    fn = "%s/rho_nlsa_scan_%d/list.txt" % (path, scan_n)
    fopen = open(fn, 'r')
    positions = []
    for i in fopen:
        print i
        h5_fn = i.split()[-1]
        print h5_fn
        
        datafile = h5py.File("%s/rho_nlsa_scan_%d/%s" % (path, scan_n, h5_fn),
                             "r")
        
        data = datafile['data']
        ch = data['SLAAR11-LMOT-M452:ENC_1_BS']
        
        print ch.keys()
        
        pos_mm = ch['data'][:].flatten()
        pos_mm = list(pos_mm)
        for item in pos_mm:
            
            positions.append(item)
            positions_all.append(item)
        
    matplotlib.pyplot.scatter(range(len(positions)), positions)
    matplotlib.pyplot.savefig("%s/positions_mm_scan_%d.png" % (path, scan_n))
    matplotlib.pyplot.close()
        # #print test.keys()
        
matplotlib.pyplot.figure(figsize=(15,5))
matplotlib.pyplot.scatter(range(len(positions_all)), positions_all)
matplotlib.pyplot.savefig("%s/positions_mm_all.png" % path)
matplotlib.pyplot.close()
        