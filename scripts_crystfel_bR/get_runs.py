# -*- coding: utf-8 -*-
import os


def get_run_ns():
    lists_original = os.popen("ls ./lists_original/*.txt").read()
    lists_original = lists_original.split("\n")[:-1]
    run_ns = []

    for list_original in lists_original:
        print(list_original)
        list_original_open = open(list_original, "r")
        for event in list_original_open:
            run_n = int(event.split("/")[9][1:5])
            run_ns.append(run_n)
        list_original_open.close()
    return run_ns


def make_run_list(out_folder, run):
    out_open = open("%s/event_list_run_%0.4d.txt" % (out_folder, run), "w")

    lists_original = os.popen("ls ./lists_original/*.txt").read()
    lists_original = lists_original.split("\n")[:-1]

    for list_original in lists_original:
        list_original_open = open(list_original, "r")
        for event in list_original_open:
            run_n = int(event.split("/")[9][1:5])
            if run_n == run:
                out_open.write(event)
        list_original_open.close()
    out_open.close()


def make_run_lists(out_folder, run_ns_unique):
    for run in run_ns_unique:
        print(run)
        make_run_list(out_folder, run)


def check(out_folder):
    n_tot = 0
    out_folder_lst = os.popen("ls %s/event*.txt" % out_folder).read()
    out_folder_lst = out_folder_lst.split("\n")[:-1]
    for j in out_folder_lst:
        tmp = open(j, "r")
        n_events = len(list(tmp))
        tmp.close()
        n_tot += n_events
    return n_tot


def make_short_lists(out_folder, out_folder_short):
    out_folder_lst = os.popen("ls %s/event*.txt" % out_folder).read()
    out_folder_lst = out_folder_lst.split("\n")[:-1]
    for k in out_folder_lst:
        k_fn = k.split("/")[-1]
        print(k_fn)
        fn_out_short = "%s/%s_1000_events.txt" % (out_folder_short, k_fn[:-4])
        os.system("touch %s" % fn_out_short)
        os.system("head -n 1000 %s > %s" % (k, fn_out_short))


if __name__ == "__main__":
    run_ns = get_run_ns()
    print("Total n. of events: ", len(run_ns))
    run_ns_unique = set(run_ns)

    out_folder = "./lists_new"
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    make_run_lists(out_folder, run_ns_unique)
    n_tot = check(out_folder)
    print("Total n. of events: ", n_tot)

    out_folder_short = "./lists_new_short"
    if not os.path.exists(out_folder_short):
        os.mkdir(out_folder_short)

    make_short_lists(out_folder, out_folder_short)
