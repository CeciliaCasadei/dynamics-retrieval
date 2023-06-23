Example of data preparation for rhodopsin data collected at SwissFEL.

- CrystFEL's indexamajig to obtain list of indexed frames in stream files.
- CrystFEL's  (0.6.3) partialator to obtain log file containing frame-specific scale and B-factor

- timing.py
  - get_ts_distribution()
  - export_unique_ID()
  - merge_scan_data()
  - select()
  
- script_read_partialator_params.m

- grep -n ‘Image filename’ fn.stream > fn.txt
- grep -n ‘Event’          fn.stream > fn.txt

- script_read_stream_line.m

- script_read_stream_params.m

- timing.py
  - export_unique_ID_ts()
  
- script_read_timestamps.m

- script_match_timestamps.m

- run_script_grab_scalable_intensities.sh (parallel run)
  of script_grab_scalable_intensities.m
  calls asuP222.m
 
- script_merge_metadata.m

- script_hklI_rescut.m

- script_hklI_redundancy.m

- script_time_sort.m

- scrpt_packing.m

- script_pack_myData_drl_scl.m

- script_generate_uniform_delay_gaussian.m

- save_full.m

START PIPELINE (run_TR-SFX_LPSA.py) IN WORKFLOWS, 
CALLING FUNCTIONS FROM dynamics_retrieval

- convert.py
- calculate_dI.py
- boost.py
ETC.