import getopt
import sys

import joblib
from pymol import cmd


def myfunc_bcolor():
    cmd.load("./6g7h_edited_nonH.pdb")
    # cmd.color('blue', selection=' (name C*)')
    cmd.hide("all")
    cmd.cartoon("tube", "all")
    cmd.set("cartoon_transparency", 0.3, "all")
    cmd.show("cartoon")
    cmd.bg_color("white")
    cmd.select(
        "sel",
        "((chain A and (resi 57 or resi 82 or resi 85 or resi 182 or resi 212 or resi 216 or resn RET) and not h.) or (chain A and (resi 407 or resi 411 or resi 415)))",
    )
    cmd.show("lines", "sel")
    cmd.show("spheres", "sel")
    cmd.set("sphere_scale", 0.10, "all")
    cmd.spectrum("b", "white_magenta")
    cmd.spectrum("b", "white_magenta", "sel")
    cmd.ray(2048, 1024)
    cmd.png("./test_bcolor.png")


def myfunc_bins():
    cmd.load("./6g7h.pdb")
    cmd.show("sticks", "all")
    cmd.bg_color("white")
    cmd.hide("all")
    cmd.select(
        "sel",
        "((chain A and (resi 212 or resi 216 or resn RET) and not h.) or (chain A and (resi 407 or resi 411 or resi 415)))",
    )
    cmd.show("lines", "sel")
    cmd.show("spheres", "sel")
    cmd.set("sphere_scale", 0.10, "all")

    cmd.load("./2.3_I_late_avg_light--dark_I_dark_avg.ccp4", "map")
    cmd.zoom("sel")

    cmd.set_view(
        "\
                     0.031440444,    0.163291126,   -0.986077130,\
                    -0.890680611,    0.452252597,    0.046491794,\
                     0.453546643,    0.876818836,    0.159659103,\
                    -2.000000000,    0.000000000,  -40.348606110,\
                    16.124301910,   44.466548920,   32.474205017,\
                    31.811149597,   48.886062622,  -20.000000000 "
    )

    cmd.isomesh("map_p", "map", 4.0, "sel", carve=2.0)
    cmd.color("cyan", "map_p")
    cmd.isomesh("map_m", "map", -4.0, "sel", carve=2.0)
    cmd.color("purple", "map_m")
    cmd.set("mesh_width", 0.5)
    cmd.set("fog_start", 0.1)
    cmd.show("mesh", "map_p")
    cmd.show("mesh", "map_m")
    cmd.ray(2048, 1024)
    cmd.png("./2.3_I_late_avg_light--dark_I_dark_avg_4p0sig.png")

    cmd.delete("map_p")
    cmd.delete("map_m")
    cmd.delete("map")


def myfunc_step(myArguments):
    try:
        optionPairs, leftOver = getopt.getopt(myArguments, "h", ["nmodes="])
    except getopt.GetoptError:
        print("Usage: ...")
        sys.exit(2)
    for option, value in optionPairs:
        if option == "-h":
            print("Usage: ...")
            sys.exit()
        elif option == "--nmodes":
            n_modes = int(value)

    print("N MODES: ", n_modes)
    t_r_p_0 = joblib.load("./t_r_p_0.jbl")
    cmd.load("./6g7h.pdb")
    cmd.color("blue", selection=" (name C*)")
    cmd.load("./6g7k.pdb")
    cmd.show("sticks", "all")
    cmd.bg_color("white")
    cmd.hide("all")
    cmd.select(
        "sel",
        "((chain A and (resi 212 or resi 216 or resn RET) and not h.) or (chain A and (resi 407 or resi 411 or resi 415)))",
    )
    cmd.show("lines", "sel")
    cmd.show("spheres", "sel")
    cmd.set("sphere_scale", 0.10, "all")

    for time in range(0, 104600, 100):
        t = t_r_p_0[time]
        cmd.load(
            "./1.8_bR_light_p_0_%d_modes_timestep_%0.6d_light--dark_I_dark_avg.ccp4"
            % (n_modes, time),
            "nlsa_map",
        )

        cmd.set_view(
            "\
		               0.306473434,    0.183651671,   -0.933994949,\
		              -0.874885261,    0.440932453,   -0.200377792,\
		               0.375029773,    0.878551126,    0.295809299,\
		               0.200000018,    2.000000000,  -43.038898468,\
		              17.967073441,   40.217617035,   33.910476685,\
		             -45.510078430,  131.587860107,  -20.000000000 "
        )
        cmd.isomesh("map_p", "nlsa_map", 4.0, "sel", carve=2.0)
        cmd.color("cyan", "map_p")
        cmd.isomesh("map_n", "nlsa_map", -4.0, "sel", carve=2.0)
        cmd.color("purple", "map_n")
        cmd.set("mesh_width", 0.3)
        cmd.set("fog_start", 0.1)
        cmd.show("mesh", "map_p")
        cmd.show("mesh", "map_n")

        cmd.pseudoatom("foo")
        cmd.set("label_size", -3)
        cmd.label("foo", "%0.1f" % t)
        cmd.set("label_position", (0, -12, 0))

        cmd.ray(2048, 1024)
        cmd.png(
            "./FRAMES_light_p_0_%d_modes_4p0sig_label/time_%0.6d_4p0sigma.png"
            % (n_modes, time)
        )

        cmd.delete("map_p")
        cmd.delete("map_n")
        cmd.delete("nlsa_map")


def myfunc_C20():
    cmd.load("./6g7h.pdb")
    cmd.color("blue", selection=" (name C*)")
    cmd.load("./6g7k.pdb")
    cmd.show("sticks", "all")
    cmd.bg_color("white")
    cmd.hide("all")
    cmd.select(
        "sel",
        "((chain A and (resi 212 or resi 216 or resn RET) and not h.) or (chain A and (resi 407 or resi 411 or resi 415)))",
    )
    cmd.show("lines", "sel")
    cmd.show("spheres", "sel")
    cmd.set("sphere_scale", 0.10, "all")

    for time in range(100, 99200, 100):
        cmd.load(
            "./map_mode_0_2_phase_6g7h/1.5_bR_light_mode_0_2_timestep_%d_light--dark_bR_dark_mode_0_avg.ccp4"
            % (time),
            "nlsa_map",
        )
        cmd.zoom("sel")
        cmd.set_view(
            "\
                         19696312,    0.989641845,   -0.079237513,\
                        955249608,    0.136545390,    0.262389272,\
                         70491034,    0.044284314,    0.961704612,\
                        799999952,    0.000000000,  -20.975885391,\
                        967073441,   40.217617035,   33.910476685,\
                      0.573089600,  109.524818420,  -20.000000000"
        )
        cmd.isomesh("map_modes_0_x_p4", "nlsa_map", 3.5, "sel", carve=2.0)
        cmd.color("cyan", "map_modes_0_x_p4")
        cmd.isomesh("map_modes_0_x_m4", "nlsa_map", -3.5, "sel", carve=2.0)
        cmd.color("purple", "map_modes_0_x_m4")
        cmd.set("mesh_width", 0.3)
        cmd.set("fog_start", 0.1)
        cmd.show("mesh", "map_modes_0_x_p4")
        cmd.show("mesh", "map_modes_0_x_m4")
        cmd.ray(2048, 1024)
        cmd.png(
            "./FRAMES_mode_0_2_phase_6g7h_3p5sig_C20_b/nlsa_modes_0_2_time_%d.png"
            % (time)
        )

        cmd.delete("map_modes_0_x_p4")
        cmd.delete("map_modes_0_x_m4")
        cmd.delete("nlsa_map")


def myfunc_penta():
    n_modes = 1
    t_r_p_0 = joblib.load("./t_r_p_0.jbl")
    cmd.load("./6g7h.pdb")
    cmd.color("lime", selection=" (name C*)")
    cmd.show("sticks", "all")
    cmd.bg_color("white")
    cmd.hide("all")
    cmd.select(
        "sel",
        "((chain A and (resi 57 or resi 82 or resi 85 resi 212 or resi 216 or resn RET) and not h.) or (chain A and (resi 407 or resi 411 or resi 415)))",
    )
    cmd.show("lines", "sel")
    cmd.show("spheres", "sel")

    cmd.set("sphere_scale", 0.10, "all")

    for time in range(0, 104600, 100):
        t = t_r_p_0[time]
        cmd.load(
            "./1.8_bR_light_p_0_%d_modes_timestep_%0.6d_light--dark_I_dark_avg.ccp4"
            % (n_modes, time),
            "nlsa_map",
        )
        print("loaded", t)
        cmd.zoom("sel")
        cmd.set_view(
            "\
     			    0.894949734,   -0.088602446,    0.437280059,\
     			    0.444157988,    0.269824058,   -0.854351640,\
			       -0.042292103,    0.958824337,    0.280833066,\
     			    0.200000018,    2.000000000,  -43.038898468,\
			       17.967073441,   40.217617035,   33.910476685,\
   			      -40.510078430,  126.587860107,  -20.000000000 "
        )

        cmd.isomesh("map_modes_0_x_p4", "nlsa_map", 4.0, "sel", carve=2.0)
        cmd.color("cyan", "map_modes_0_x_p4")
        cmd.isomesh("map_modes_0_x_m4", "nlsa_map", -4.0, "sel", carve=2.0)
        cmd.color("purple", "map_modes_0_x_m4")
        cmd.set("mesh_width", 0.3)
        cmd.set("fog_start", 0.1)
        cmd.show("mesh", "map_modes_0_x_p4")
        cmd.show("mesh", "map_modes_0_x_m4")
        cmd.clip("slab", 20)
        cmd.clip("far", +6)

        cmd.pseudoatom("foo")
        cmd.set("label_size", -2)
        cmd.label("foo", "%0.1f" % t)
        cmd.set("label_position", (+4, -15, +5))

        cmd.ray(2048, 1024)
        cmd.png(
            "./FRAMES_light_p_0_%d_modes_4p0sig_penta/time_%0.6d_4p0sigma.png"
            % (n_modes, time)
        )

        cmd.delete("map_modes_0_x_p4")
        cmd.delete("map_modes_0_x_m4")
        cmd.delete("nlsa_map")


def myfunc_cartoon():
    n_modes = 1
    t_r_p_0 = joblib.load("./t_r_p_0.jbl")
    cmd.load("./6g7h.pdb")
    cmd.hide("all")
    cmd.cartoon("tube", "all")
    cmd.set("cartoon_transparency", 0.7, "all")
    cmd.set("cartoon_side_chain_helper", "on")
    cmd.show("cartoon")
    cmd.spectrum("resi", "rainbow", "all")
    cmd.bg_color("white")
    cmd.select(
        "sel",
        "((chain A and (resi 47-57 or resi 82 or resi 85 or resi 182 or resi 206 or resi 208 or resi 212 or resi 216 or resn RET) and not h.) or (chain A and (resi 407 or resi 411 or resi 415)))",
    )
    cmd.show("lines", "sel")
    cmd.show("spheres", "sel")
    cmd.set_view(
        "\
                     0.711140990,    0.037647203,   -0.702040076,\
                    -0.699935138,   -0.055949945,   -0.712014735,\
                    -0.066083595,    0.997723699,   -0.013440915,\
                     0.000000000,    1.000000000,  -62.475360870,\
                    17.146160126,   39.064826965,   35.888786316,\
                    17.158218384,  107.792404175,  -20.000000000  "
    )

    cmd.set("sphere_scale", 0.08, "all")

    for time in range(0, 104600, 100):
        t = t_r_p_0[time]
        cmd.load(
            "./1.8_bR_light_p_0_%d_modes_timestep_%0.6d_light--dark_I_dark_avg.ccp4"
            % (n_modes, time),
            "nlsa_map",
        )
        print("loaded", t)
        cmd.zoom("sel")
        cmd.isomesh("map_modes_0_x_p4", "nlsa_map", 4.0, "all", carve=5.0)
        cmd.color("cyan", "map_modes_0_x_p4")
        cmd.isomesh("map_modes_0_x_m4", "nlsa_map", -4.0, "all", carve=5.0)
        cmd.color("purple", "map_modes_0_x_m4")
        cmd.set("mesh_width", 0.3)
        cmd.set("fog_start", 0.1)
        cmd.show("mesh", "map_modes_0_x_p4")
        cmd.show("mesh", "map_modes_0_x_m4")
        cmd.pseudoatom("foo")
        cmd.set("label_size", -5)
        cmd.label("foo", "%0.1f" % t)
        cmd.set("label_position", (0, -25, 0))

        cmd.ray(2048, 1024)
        cmd.png(
            "./FRAMES_light_p_0_%d_modes_4p0sig_cartoon/time_%0.6d_4p0sigma.png"
            % (n_modes, time)
        )

        cmd.delete("map_modes_0_x_p4")
        cmd.delete("map_modes_0_x_m4")
        cmd.delete("nlsa_map")


cmd.extend("myfunc_bins", myfunc_bins)
cmd.extend("myfunc_step", myfunc_step)
cmd.extend("myfunc_C20", myfunc_C20)
cmd.extend("myfunc_penta", myfunc_penta)
cmd.extend("myfunc_cartoon", myfunc_cartoon)
cmd.extend("myfunc_bcolor", myfunc_bcolor)


# myfunc_step(sys.argv[1:])
# myfunc_cartoon()
myfunc_penta()
