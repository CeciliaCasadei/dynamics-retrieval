#!/bin/bash
module load PyMOL
#pymol -c cmd_pymol_python_bR.py
#pymol -c cmd_pymol_python_bR.py --inputMode 6
pymol -c cmd_pymol_python_rho.py --chainID B --sig 2.5
