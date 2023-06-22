# Dynamics Retrieval

Dynamics retrieval (SSA, LPSA, NLSA) methods with application to time-resolved serial crystallography data and other (synthetic, climate).

References:

> Casadei, C. M. et al., Structural Dynamics (2022),
> Dynamics retrieval from stochastically weighted incomplete data by low-pass spectral analysis.
> https://doi.org/10.1063/4.0000156

> Casadei, C. M. et al., Structural Dynamics (2023),
> Low-pass spectral analysis of time-resolved serial crystallography data.
> https://doi.org/10.1063/4.0000178

## Installation

Start by cloning the code.

    git clone https://github.com/CeciliaCasadei/dynamics-retrieval.git
    cd dynamics-retrieval

Using conda is recommended to install dependencies. A new conda environment can
be created with

    conda env create -f environment.yml

After this, install the package:

    pip install -e .[dev]

Many workflows currently require editing the source code, so installing in developer
mode (`-e`) is recommended.

## Testing

To test:

    cd workflows
    python test_package.py
    
## Workflows

The LPSA and NLSA code are contained in the `dynamics_retrieval` package.
However significant pre-processing is required to prepare data for analysis.
Preparation and analysis scripts are provided in the `scripts*` directories,
which can be customized to your application. Code for LPSA and NLSA analysis is
contained in the library, with wrappers calling the functions within
`workflows` directory.

### TR-SFX Workflow

This is the general workflow used for serial crystallography. Scripts for bovine
rhodopsin (rho) and bacteriorhodopsin (bR) are provided. Bacteriorhodopsin
TR-SFX data can be found on [zenodo](https://doi.org/10.5281/zenodo.7896581).
The general flow is as follows:

- `scripts_crystfel_*`
  - Use CrystFEL to process TR-SFX data to produce stream files 
    with indexed intensities (indexamajig, ambigator)
    & a list of scale factors (partialator).
  - Calculate merging statistics, 
    to e.g. estimate the desired high-resolution cutoff.
- `scripts_data_reduction_*`
  - Start with streams, scaling factors, and space group (eg asuP6_3.m)
  - Process
    - Extract reflection intensities for each frame from the stream
    - Apply scale factors for each frame
    - Apply symmetry transformations
    - Add timing info for each frame
    - Filter for desired timing distribution
  - Output data matrix (1 column per frame)
- `workflows`
  - `run_TR-SFX_LPSA.py` runs scripts for dynamics retrieval
  - produces reconstructed reflection intensities for each timestep
- `scripts_make_maps`
  - Converts output to mtz for use in phenix
- `scripts_map_analysis`
  - Integrate difference density around a feature of interest

### Settings files

Settings are currently passed via a python module consisting of global
variables. Examples are in `workflows/settings*.py`.
