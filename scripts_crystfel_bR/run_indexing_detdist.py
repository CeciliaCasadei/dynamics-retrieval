# -*- coding: utf-8 -*-
import os

for i in range(57, 58):
    print(i)
    list_dir = "/das/work/p17/p17491/Cecilia_Casadei/NLSA/data_bR_2/indexing/event_lists/lists_new_short"
    fn = "%s/event_list_run_%0.4d_1000_events.txt" % (list_dir, i)
    if os.path.exists(fn):
        print(fn)
        os.system("./run_indexing_detdist.sh %0.4d" % i)
