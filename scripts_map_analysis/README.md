# Map Analysis

1. Run `check_PDB_limits.m` and modify `process_maps.sh` step 2 accordingly.

2. Activate cctbx environment (for PSI ra cluster):
   ```
   module use unstable
   module load cctbx
   source $CCTBX_PREFIX/conda/bin/activate $CCTBX_PREFIX/conda
   ```
   
3. For each map: 
   step 1 to 4 in `process_maps.sh`

   1. Call `mapmask` (CCP4) to extend the initial map to cover entire cell.
   2. Call `maprot` (CCP4) to translate map to cartesian coos.
   3. Call `map_to_h5.py` (import iotbx) to convert map to HDF5 format.
   4. Call `mapdump` (CCP4) to extract info on original and cartesian map, logged to txt file.

   To run 3 on a computing node use
   ```
   sbatch -p day run_process.sh
   ```

4. `maptool.m`

   - 4’871 atoms chain A+B with retinals
   - 2’553 grid points at step 0.2A in a sphere of radius 1.7A around each atom   
   - Calculate grid coords for each atom: 
     - X: 4’871 x 2’553 -> reshape to 12’435’663 x 1
     - Y: 4’871 x 2’553 -> reshape to 12’435’663 x 1
     - Z: 4’871 x 2’553 -> reshape to 12’435’663 x 1
   - Extract map grid params for each axis from `<fn>_XYZ_info.dat`
   - and make meshgrid `gY`, `gZ`, `gX` with same grid pts as map in HDF5 format.
   - Map (in h5) is defined on `gY`, `gZ`, `gX`.
   - Interpolate on `Y`, `Z`, `X`.
   - Reshape to 4’871 x 2’553.
   - Convert electron density values to sigma values.
   - Repeat for each map and save `mapd0`.

5. `maptool_getmeanden.py`

   - Load mapd0 in py:
     - 2’553 x 4’871 x 2’115
     - n_grid_pts_per_atom x n_atoms x n_maps

   - Loop on sigmacutoff values:

     - Set to zero all densities below `sigmacutoff`.
     - For each atom, get mean positive density over all grid points in the sphere surrounding it.

     - Set to zero all densities above `-sigmacutoff.`
     - For each atom, get mean negative density over all grid points in the - sphere
       surrounding it.

   - Repeat for each map.

   - Final matrices: 
     - meanposden: n_atoms x n_maps (for each mode)
     - meannegden: n_atoms x n_maps (for each mode)
   
6. `plot_densities.py`