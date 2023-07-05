#!/bin/bash
module load phenix
for i in *mtz; do cctbx.patterson_map $i; done
