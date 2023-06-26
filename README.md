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
  - See details for [bR](scripts_data_reduction_bR/README.md),
    [rhodopsin](scripts_data_reduction_rho/README.md)
  - Start with streams, scaling factors, and space group (eg asuP6_3.m)
  - Process
    - Extract reflection intensities for each frame from the stream
    - Apply scale factors for each frame
    - Apply symmetry transformations
    - Add timing info for each frame
    - Filter for desired timing distribution (eg uniform timepoints)
  - Output data matrix (1 column per frame)
- `workflows`
  - `run_TR-SFX_LPSA.py` runs scripts for dynamics retrieval
  - produces reconstructed reflection intensities for each timestep
- `scripts_make_maps` 
  - Converts output to mtz for use in phenix
- [`scripts_map_analysis`](scripts_map_analysis/README.md)
  - Integrate difference density around a feature of interest


## Conventions

The following naming conventions can be helpful in understanding the code. Variable
names often reflect the mathematical notation used in the papers.

- `x`, the main input with each frame as one column (*m* reflections ⨯ *S* timesteps)
  - For SFX typical sizes would be *m=10^4*, *S=10^5*
- `q`, the concatenation number. Number of frames in *x* that get concatenated together
  to form supervectors in *X*. Should be odd.
  - *q* needs to be optimized, but values on order of *10^4* worked for SFX
- `X`, the superframe matrix (*qm* reflections ⨯ *S* timesteps). Not stored explicitely.
- `F` or Φ, the matrix of harmonic functions
- `jmax` or `f_max`: maximum harmonic frequency to keep
- `p`, number of frames in the reconstructed *X* to average to make reconstructed *x*.
  *p = 0* takes a single frame (for performance); *2p+1 = q* then all frames are
  averaged.
- 20, number of modes to save (hard coded in several functions)


### Temporary files

To allow workflows to be resumed, large objects are often saved to joblib (`.jbl`) files
and then loaded during later steps. Files are expected to be saved to various fixed
paths relative to a temporary directory, the `results_path`.

Examples:
- `input_data_sparsity_*.jbl`: *x* matrix (*s* frames ⨯ *m* reflections) giving
  intensities for each (h,k,l)
- `dT.jbl`
- `input_data_mask_sparsity*.jbl`: masks measured values of *x*
- `F_on.jbl`: Phi matrix with LPSA harmonic functions, orthonomalized

### Settings files

Settings are currently passed via a python module consisting of global variables.
Examples are in `workflows/settings*.py`. Typical variables include the `results_path`,
`S`, `q`, etc.
