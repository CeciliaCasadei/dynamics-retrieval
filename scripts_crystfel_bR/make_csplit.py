# -*- coding: utf-8 -*-
import os

import h5py
import joblib
import numpy


def f_class1_class4():
    os.system('grep "Image filename" all_detwinned.stream > img_fn.txt')
    os.system('grep "Event" all_detwinned.stream > event.txt')

    img_fn = list(open("img_fn.txt", "r"))
    event_file = list(open("event.txt", "r"))
    n_events = len(img_fn)

    out_fn = open("csplit.dat", "w")
    for i in range(n_events):
        print(i)
        if "class1" in img_fn[i]:
            out_fn.write(
                "%s //%s class1\n"
                % (img_fn[i][16:].strip("\n"), event_file[i][9:].strip("\n"))
            )
        elif "class4" in img_fn[i]:
            out_fn.write(
                "%s //%s class4\n"
                % (img_fn[i][16:].strip("\n"), event_file[i][9:].strip("\n"))
            )
        else:
            print("issue")
    out_fn.close()


def f_partiality_model():

    uniqueID_dark_lst = list(open("./uniqueID_dark.txt", "r"))
    uniqueID_light_lst = list(open("./uniqueID_light.txt", "r"))
    file_mat = (
        "./data_bR_light_selected_int_unifdelay_SCL_rescut_nS133115_nBrg39172.mat"
    )
    f = h5py.File(file_mat, "r")
    timestamps = f["/t_uniform"]
    timestamps = numpy.asarray(timestamps).flatten()
    print("timestamps: ", timestamps)
    test = timestamps[numpy.argwhere(timestamps > 0)].flatten()
    test = test[numpy.argwhere(test < 1000)]
    print(test.shape, "frames in [0, 1000]fs")

    myDict = {}
    myDict["uniqueIDs_dark"] = uniqueID_dark_lst
    myDict["uniqueIDs_light"] = uniqueID_light_lst
    myDict["timestamps"] = timestamps
    joblib.dump(myDict, "dict_uniqueIDs_timestamps.jbl")

    os.system('grep "Image filename" all_detwinned.stream > img_fn.txt')
    os.system('grep "Event" all_detwinned.stream > event.txt')

    img_fn = list(open("img_fn.txt", "r"))
    event_file = list(open("event.txt", "r"))
    n_events = len(img_fn)

    out_fn = open("csplit_partialitymodeling.dat", "w")
    for i in range(n_events):
        image_fn = img_fn[i][16:].strip("\n")
        event_n = event_file[i][9:].strip("\n")
        label = image_fn + "_event_" + event_n

        # print label
        if "class1" in image_fn:
            if label + "\n" in uniqueID_light_lst:
                print("light - IN")
                label = label.split("/")[-1]
                out_fn.write("%s //%s %s\n" % (image_fn, event_n, label))
        elif "class4" in image_fn:
            if label + "\n" in uniqueID_dark_lst:
                print("dark - IN")
                label = label.split("/")[-1]
                out_fn.write("%s //%s %s\n" % (image_fn, event_n, label))
        else:
            print("issue")
    out_fn.close()


# f_select_stream():
# os.system('grep -n "Image filename" all_detwinned.stream > img_fn_line_ns.txt')
# os.system('grep -n "Event"          all_detwinned.stream > event_line_ns.txt' )
# img_fn_line_ns = list(open('img_fn_line_ns.txt', 'r'))
# event_line_ns  = list(open('event_line_ns.txt',  'r'))
# N = len(img_fn_line_ns)
# line_ns = []
# img_fns = []
# event_ns = []
# for i in range(N):
#     print i
#     mysplit = img_fn_line_ns[i].split(':')
#     line_n = mysplit[0]
#     fn = mysplit[-1][1:].strip('\n')
#     mysplit = event_line_ns[i].split(':')
#     event = mysplit[-1][3:].strip('\n')
#     line_ns.append(line_n)
#     img_fns.append(fn)
#     event_ns.append(event)
# joblib.dump(line_ns,  './line_ns.jbl')
# joblib.dump(img_fns,  './img_fns.jbl')
# joblib.dump(event_ns, './event_ns.jbl')

# line_ns  = joblib.load('./line_ns.jbl')
# img_fns  = joblib.load('./img_fns.jbl')
# event_ns = joblib.load('./event_ns.jbl')
# frames_to_include = list(open('csplit_partialitymodeling.dat', 'r'))
# n_frames_to_include = len(frames_to_include)
# labels_frames_to_include = []
# for i in range(n_frames_to_include):
#     label = frames_to_include[i].split()[-1].strip('\n')
#     #print label
#     labels_frames_to_include.append(label)

# print 'done'
# #all_detwinned = list(open('all_detwinned.stream', 'r'))
# N = len(line_ns)
# n = 0
# out_f = open('cmd.sh', 'w')
# for i in range(5):#N):
#     print i
#     line_n = line_ns[i]
#     img_fn = img_fns[i]
#     event_n = event_ns[i]
#     label = img_fn.split('/')[-1] + '_event_' + event_n
#     #print label

#     if label in labels_frames_to_include:
#         print 'in'

#         n = n+1
#         start_line = int(line_n)-1
#         end_line = int(line_ns[i+1])-2
#         out_f.write('echo %d\n'%n)
#         out_f.write('sed -n "%d,%dp" all_detwinned.stream >> selected_detwinned.stream\n'%(start_line, end_line))
#         #out_f.write('sed -i selected_detwinned.stream -re "%d,%dd"\n'%(start_line, end_line))
#         #os.system('sed -n "%d,%dp" all_detwinned.stream >> selected_detwinned.stream'%(start_line, end_line))
# print n
# out_f.close()


# f_class1_class4()
f_partiality_model()
