import getopt
import os
import sys

import joblib
from pymol import cmd
from pymol.cgo import *


def myfunc_sphere():

    cmd.load("./bov_nlsa_refine_96_chainA.pdb")
    cmd.color("blue", selection=" (name C*)")
    cmd.color("lightblue", selection=" (name N*)")
    cmd.show("sticks", "all")
    cmd.bg_color("white")
    cmd.hide("all")
    cmd.select("sel", "resn RET or resi 296")
    cmd.show("lines", "sel")
    cmd.show("spheres", "sel")
    cmd.set("sphere_scale", 0.10, "all")

    spherelist = [
        COLOR,
        0.5,
        1,
        1,
        ALPHA,
        0.5,
        SPHERE,
        9.69,
        25.67,
        26.92,
        1.70,  # Positive peak
    ]

    cmd.load_cgo(spherelist, "segment", 1)

    cmd.set_view(
        "\
                           0.080371954,    0.806613684,   -0.585589767,\
                          -0.930780470,   -0.149466500,   -0.333627909,\
                          -0.356634021,    0.571868539,    0.738767087,\
                           0.000000000,    0.000000000,  -38.521888733,\
                           9.948150635,   28.112447739,   29.300647736,\
                          30.370950699,   46.672828674,  -20.000000000 "
    )
    cmd.move("y", -1)
    cmd.move("x", -3)
    cmd.ray(2048, 1024)
    cmd.png("./rho_spheres_1p7A_positive_peak.png")


def myfunc_step(myArguments):
    try:
        optionPairs, leftOver = getopt.getopt(
            myArguments, "h", ["nModes=", "chainID=", "sig="]
        )
    except getopt.GetoptError:
        print("Usage: ...")
        sys.exit(2)
    for option, value in optionPairs:
        if option == "-h":
            print("Usage: ...")
            sys.exit()
        elif option == "--nModes":
            nmodes = int(value)
        elif option == "--chainID":
            chainID = str(value)
        elif option == "--sig":
            sig = float(value)

    t_r_p_0 = joblib.load("./t_r_p_0.jbl")

    cmd.load("./swissfel_combined_dark_frac_0.01_0.19_ref_chain%s.pdb" % chainID)
    cmd.color("blue", selection=" (name C*)")
    cmd.color("lightblue", selection=" (name N*)")
    cmd.show("sticks", "all")
    cmd.bg_color("white")
    cmd.hide("all")

    # cmd.select('sel', 'resn RET or resi 113 or resi 268 or resi 191 or resi 118 or resi 265')
    # cmd.select('sel', 'resi 113 or resi 118 or resi 186 or resi 207 or resi 212 or resi 265 or resi 268 or resi 296 or resn RET')
    cmd.select("sel", "resn RET")
    cmd.show("lines", "sel")
    cmd.show("spheres", "sel")
    cmd.set("sphere_scale", 0.10, "all")

    map_folder = "."
    out_folder = "./FRAMES_%d_modes_%sp%ssig_RET%s" % (
        nmodes,
        str(sig)[0],
        str(sig)[2],
        chainID,
    )

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    for time in range(0, 43100, 100):
        t = t_r_p_0[time]
        cmd.load(
            "%s/1.8_extracted_Is_p_0_%d_modes_timestep_%0.6d_light--dark_swissfel_combined_dark_frac_0.01_0.19_0.80.ccp4"
            % (map_folder, nmodes, time),
            "nlsa_map",
        )
        cmd.zoom("sel")

        # RET A
        if chainID == "A":
            cmd.set_view(
                "\
                           0.080371954,    0.806613684,   -0.585589767,\
                          -0.930780470,   -0.149466500,   -0.333627909,\
                          -0.356634021,    0.571868539,    0.738767087,\
                           0.000000000,    0.000000000,  -38.521888733,\
                           9.948150635,   28.112447739,   29.300647736,\
                          30.370950699,   46.672828674,  -20.000000000 "
            )
            cmd.move("y", -1)
            cmd.move("x", -3)

        # RET B
        else:
            cmd.set_view(
                "\
                            0.203187689,    0.868704140,   -0.451742560,\
                            0.946889222,   -0.056905121,    0.316471487,\
                            0.249215886,   -0.492051393,   -0.834131420,\
                            0.000000000,    0.000000000,  -39.938152313,\
                          -20.810651779,   28.454404831,   46.120998383,\
                           31.775022507,   48.101264954,  -20.000000000 "
            )
            cmd.move("x", -3)

        cmd.isomesh("map_modes_0_x_p", "nlsa_map", sig, "sel", carve=2.0)
        cmd.color("cyan", "map_modes_0_x_p")
        cmd.isomesh("map_modes_0_x_m", "nlsa_map", -sig, "sel", carve=2.0)
        cmd.color("purple", "map_modes_0_x_m")
        cmd.set("mesh_width", 0.3)
        # cmd.set('fog_start', 0.1)
        cmd.show("mesh", "map_modes_0_x_p")
        cmd.show("mesh", "map_modes_0_x_m")

        cmd.pseudoatom("foo")
        cmd.set("label_size", -3)
        cmd.label("foo", "%0.1f" % t)
        cmd.set("label_position", (0, -12, 0))

        cmd.ray(2048, 1024)
        cmd.png("%s/%d_modes_time_%0.6d_t.png" % (out_folder, nmodes, time))

        cmd.delete("map_modes_0_x_p")
        cmd.delete("map_modes_0_x_m")
        cmd.delete("nlsa_map")


def myfunc_bin(myArguments):
    try:
        optionPairs, leftOver = getopt.getopt(myArguments, "h", ["chainID=", "sig="])
    except getopt.GetoptError:
        print("Usage: ...")
        sys.exit(2)
    for option, value in optionPairs:
        if option == "-h":
            print("Usage: ...")
            sys.exit()
        elif option == "--chainID":
            chainID = str(value)
        elif option == "--sig":
            sig = float(value)

    cmd.load("./swissfel_combined_dark_frac_0.01_0.19_ref_chain%s.pdb" % chainID)
    cmd.color("blue", selection=" (name C*)")
    cmd.color("lightblue", selection=" (name N*)")
    cmd.show("sticks", "all")
    cmd.bg_color("white")
    cmd.hide("all")
    # cmd.select('sel', '((chain A and (resi 212 or resi 216 or resn RET) and not h.) or (chain A and (resi 407 or resi 411 or resi 415)))')
    cmd.select("sel", "resn RET or resi 296")
    cmd.show("lines", "sel")
    cmd.show("spheres", "sel")
    cmd.set("sphere_scale", 0.10, "all")

    map_folder = "."
    out_folder = "."
    time_bin_labels = ["early", "late"]
    for time_bin_label in time_bin_labels:
        cmd.load(
            "%s/1.8_I_%s_avg_light--dark_swissfel_combined_dark_frac_0.01_0.19_0.80.ccp4"
            % (map_folder, time_bin_label),
            "mymap",
        )
        cmd.zoom("sel")

        # RET A
        if chainID == "A":
            cmd.set_view(
                "\
                           0.080371954,    0.806613684,   -0.585589767,\
                          -0.930780470,   -0.149466500,   -0.333627909,\
                          -0.356634021,    0.571868539,    0.738767087,\
                           0.000000000,    0.000000000,  -38.521888733,\
                           9.948150635,   28.112447739,   29.300647736,\
                          30.370950699,   46.672828674,  -20.000000000 "
            )
            cmd.move("y", -1)
            cmd.move("x", -3)

        # RET B
        else:
            cmd.set_view(
                "\
                            0.203187689,    0.868704140,   -0.451742560,\
                            0.946889222,   -0.056905121,    0.316471487,\
                            0.249215886,   -0.492051393,   -0.834131420,\
                            0.000000000,    0.000000000,  -39.938152313,\
                          -20.810651779,   28.454404831,   46.120998383,\
                           31.775022507,   48.101264954,  -20.000000000 "
            )
            cmd.move("x", -3)

        cmd.isomesh("map_plus", "mymap", sig, "sel", carve=2.0)
        cmd.color("cyan", "map_plus")
        cmd.isomesh("map_minus", "mymap", -sig, "sel", carve=2.0)
        cmd.color("purple", "map_minus")
        cmd.set("mesh_width", 0.3)
        cmd.show("mesh", "map_plus")
        cmd.show("mesh", "map_minus")
        cmd.ray(2048, 1024)

        cmd.png(
            "%s/1.8_I_%s_avg_light--dark_I_dark_avg_chain%s_%.2fsig.png"
            % (out_folder, time_bin_label, chainID, sig)
        )

        cmd.delete("map_plus")
        cmd.delete("map_minus")
        cmd.delete("mymap")


cmd.extend("myfunc_step", myfunc_step)
cmd.extend("myfunc_bin", myfunc_bin)
cmd.extend("myfunc_sphere", myfunc_sphere)

myfunc_step(sys.argv[1:])
# myfunc_sphere()
# myfunc_bin(sys.argv[1:])
