#!/bin/bash
module load PyMOL
#pymol -c test.py
pymol -c cmd_pymol_python_rho.py --inputMode 1 --chainID A --sig 3.5
