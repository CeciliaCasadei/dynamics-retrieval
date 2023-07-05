# -*- coding: utf-8 -*-
import os

from . import calculate_distances_utilities

if __name__ == "__main__":

    ### Concatenation followed by distance calculation (optimised - dense)
    # calculate_distances_utilities.concatenate_backward()
    # calculate_distances_utilities.calculate_D_N_v_optimised_dense()

    ### Concatenation followed by distance calculation (optimised - sparse)
    # calculate_distances_utilities.concatenate_backward()
    # calculate_distances_utilities.calculate_D_N_v_optimised_sparse()

    ### Calculate d_sq followed by D_sq (non parallel - dense)
    # calculate_distances_utilities.calculate_d_sq_dense()
    # calculate_distances_utilities.calculate_D_sq()
    calculate_distances_utilities.sort_D_sq()

    ### Calculate d_sq followed by D_sq (non parallel - sparse)
    # calculate_distances_utilities.calculate_d_sq_sparse()
    # calculate_distances_utilities.calculate_D_sq()
    # calculate_distances_utilities.sort_D_sq()

    ### Calculate d_sq followed by D_sq (parallel - dense)
    # calculate_distances_utilities.calculate_d_sq_dense()
    # os.system('sbatch run_parallel.sh')
    # calculate_distances_utilities.merge_D_sq()
    # calculate_distances_utilities.sort_D_sq()

    ### Calculate d_sq followed by D_sq (parallel - sparse)
    # calculate_distances_utilities.calculate_d_sq_sparse()
    # os.system('sbatch run_parallel.sh')
    # calculate_distances_utilities.merge_D_sq()
    # calculate_distances_utilities.sort_D_sq()

    ### Concatenation followed by distance calculation (standard - dense)
    # calculate_distances_utilities.concatenate_backward()
    # calculate_distances_utilities.calculate_D_N_v_dense()

    ### Concatenation followed by distance calculation (standard - sparse)
    # calculate_distances_utilities.concatenate_backward()
    # calculate_distances_utilities.calculate_D_N_v_sparse()
