# WRF perturbation studies at LES resolution

This project contains modifications to WRF code, plus runtime and analysis code to investigate the response of WRF to temperature and humidity perturbations. 
Broadly the idea is follow the linear response function (LRF) approach of [Kuang (2010)](https://doi.org/10.1175/2009JAS3260.1) and to run WRF at large eddy 
simulation (LES) resolutions.

## Directory structure

- `WRF` - WRF source files modified for this project.
- `scripts` - Scripts organised by language.
- `runtime` - Data files used for running WRF.
- `analysis` - Jupyter notebooks and python code used for analysis.

## Installation

The script `scripts/sh/install_wrf.sh` makes backup copies of the original version of modified files, then makes symlinks in a WRF installation to the modified 
versions. The modified files are based on code from the CEOCMS version of WRF designed to run on gadi, version 4.1.4. To install and modify WRF in a directory 
`<WRFDIR>` when this project is cloned to `<GITDIR>`, use:

```
cd <WRFDIR>
git clone -b V4.1.4 git@github.com:coecms/WRF.git
cd <GITDIR>/wrf_lrf_les/WRF/v4.1.4/
../scripts/sh/install_wrf.sh <WRFDIR>/WRF/
```

## WRF compilation

In this project the `em_quarter_ss` ideal case is used as the base case. Compile it using the following:

```
cd <WRFDIR>/WRF/WRFV3/
./run_compile --clean --compile_case em_quarter_ss
```

To check compilation, look at the end of the most recent `compile_job.` file.

## Scripts

The following scripts are included in this repository in the `scripts` directory:

- Main scripts:
 	- `sh/install_wrf.sh` - install the WRF modifications in this repository into a WRF codebase.
	- `sh/setup_wrf_run.sh` - make a new runtime directory with all files required for a WRF run.
	- `sh/move_wrf_output.sh` - move all WRF output, settings, and lots for a run to a given directory.
	- `sh/run_ideal.sh` - run the WRF `ideal.exe` to prepare model.
	- `sh/run_wrf.sh` - run the WRF model in parallel.
	- `sh/extract_WRF_vars_parallel.sh` - extract WRF variables for analysis in parallel.

- Helper scripts:
	- `python/extract_WRF_vars.py` - extract, vertically interpolate and average WRF variables.
	- `sh/extract_WRF_variables.sh` - call the python extraction script for WRF variables.

## Analysis

The `analysis` directory contains Jupyter notebooks and python modules to analyse WRF output:

- Jupyter notebooks:
	- `eta_levels.ipynb` - calculate WRF vertical level (eta-level) settings for desired vertical heights in the initial model setup.
	- `perturbation_analysis.ipynb` - the main notebook for perturbation analysis.
- Python modules:
	- `modules.rcemip_profile` - code to calculate RCEMIP vertical profiles, as per [Wing et al., 2018](https://doi.org/10.5194/gmd-11-793-2018).
	- `modules.wrf_perturbation` - main code for WRF perturbation analysis.
	- `modules.wrf_profile` - code to calculate WRF profiles as per WRF's initialisation of ideal cases.
	
## How to run 

Assuming this respository is cloned to `~/git`:

1. [Install WRF and modifications](#installation)
2. [Compile WRF](#compilation)
3. Copy runtime files to a new directory using `~/git/wrf_lrf_les/scripts/sh/setup_wrf_run.sh`. 
4. `cd` to the runtime directory.
5. Edit `namelist.input` to set WRF settings.
6. Edit `run_wrf.sh` to set job settings.
7. Run `./run_ideal.sh`.
7. Use `ncview` to check that the `wrfinput` file is correct.
8. Submit the job using `qsub run_wrf.sh`.
9. Make directory for output and copy output there using `move_wrf_output.sh`.
10. `cd` to output directory.
11. Extract variables in parallel using `qsub ~/git/wrf_lrf_les/scripts/sh/extract_WRF_vars_parallel.sh`.
12. Edit notebook `perturbation_analysis.ipynb` to point to output and run to analyse perturbations.

## Modifications to WRF

### New namelist options

Under `rce_control`:
- `ssttsk` - sea surface temperature for initial state (K).
- `use_light_nudging` - use light nudging for wind fields?
- `ideal_evap_flag` - use ideal evaporation?
- `surface_wind` - surface wind value for initial state (m s-1).
- `const_rad_cooling` - use prescribed, constant radiative cooling profile as per [Herman and Kuang (2013)](https://doi.org/10.1002/jame.20037)?
- `relax_uv_winds` - relax U and V winds to target profiles?
- `wind_relaxation_time` - wind relaxation time (s).
- `relax_u_profile_file` - filename for target U wind profile.
- `relax_v_profile_file` - filename for target V wind profile.

Under `lrf_control`:
- `perturb_t` - perturb temperature (theta)?
- `perturb_q` - perturb moisture (QV)?  
- `k_pert` - the 1-based vertical level index to apply perturbations to.
- `TtendAmp` - amplitude of theta perturbation (K day-1).
- `QtendAmp` - amplitude of QV perturbation (kg kg-1 day-1).

### General notes

* Many of the changes to the code were adapted from code changes made by Yi-Ling Hwong, Maxime Colin, David Fuchs, or Nidhi Nishant for previous versions of WRF. I 
have updated the code and applied it to v4.1.4. Previously, perturbations were applied by modifying the humidity, temperature, or wind tendencies for the planetary 
boundary layer (PBL) scheme; at LES resolution the PBL scheme will be turned off, so I have made new tendency variables that are treated in the same way to other 
tendencies in the model. Note that the perturbations are added **without** explicitly hydrostatically rebalancing the model. 

* The radiative cooling profile is prescribed when `const_rad_cooling == 1`. In this case the radiation driver is called, and immediately afterwards the theta tendancy due to radiation is prescribed. Note this may affect the accuracy of the following variables which are computed by the radiation driver: `COSZEN, CLDFRA, 
SWDOWN, GLW, ACSWUPT, ACSWUPTC, ACSWDNT, ACSWDNTC, ACSWUPB, ACSWUPBC, ACSWDNB, ACSWDNBC, ACLWUPT, ACLWUPTC, ACLWUPB, ACLWUPBC, ACLWDNB, ACLWDNBC, SWUPT, SWUPTC, 
SWDNT, SWDNTC, SWUPB, SWUPBC, SWDNB, SWDNBC, LWUPT, LWUPTC, LWUPB, LWUPBC, LWDNB, LWDNBC, OLR`.

### Specific changes

Note that theta means potential temperature (K), QV means water vapour mixing ratio (kg kg-1), U and V are horizontal winds (m s-1). 'New tendencies' refers to the 
new tendency variables introduced specifically to be separate from any other scheme (PBL, radiation, etc), which are called `RTHFORCETEN` (theta), `RQVFORCETEN` 
(QV), `RUFORCETEN` (U) and `RVFORCETEN` (V) in the model code.

Changes are made to the following files:

- `WRFV3/Registry/Registry.EM_COMMON`:
	- made `RTHRATEN` (potential temperature tendency due to radiation scheme) an output variable.
	- added new grid variables:
		- `RTHFORCETEN`, `RQVFORCETEN` - tendency due to perturbation forcing in U and V respectively.
		- `RUFORCETEN`, `RVFORCETEN` - tendency due to wind relaxation in U and V respectively.
		- `RELAX_U_TARGET_PROFILE`, `RELAX_V_TARGET_PROFILE` - the target U and V wind profiles for wind relaxation.
	- added new namelist options. 
- `WRFV3/dyn_em/Makefile` - added compilation rules for `module_nudging` and `module_LRF`.
- `WRFV3/dyn_em/module_LRF.F` - new module containing the following functions:
	- `force_LRF`: force vertical temperature and moisture tendencies.
	- `relax_winds_to_profile`: do wind relaxation to target profile.
	- `read_profile_file`: read a target profile from a file.
	- `read_target_profiles`: read U and V target profile files.
- `WRFV3/dyn_em/module_big_step_utilities_em.F` - updated function  `phy_prep_part2` to take new tendencies and decouple them from mass points.
- `WRFV3/dyn_em/module_diffusion_em.F` - updated function `phy_bc` to accept new wind tendencies and set them in boundary regions.
- `WRFV3/dyn_em/module_em.F` - updated function `calculate_phy_tend` to accept new tendencies and couple them to mass points.
- `WRFV3/dyn_em/module_first_rk_step_part1.F` - added new processing steps:
	- if `const_rad_cooling` is set prescribe the radiative cooling profile as per [Herman and Kuang (2013)](https://doi.org/10.1002/jame.20037).
	- pass ideal evaporation and surface wind options to `surface_driver` function.
	- assign perturbations tendencies to theta, QV if forcing required.
	- load wind profiles and assign relaxation tendencies to U and V if required.
- `WRFV3/dyn_em/module_first_rk_step_part2.F` - updated function calls:
	- updated call to `calculate_phy_tend` to pass new tendencies.
	- updated call to `phy_bc` to pass new (wind) tendencies.
	- updated call to `update_phy_ten` to pass new tendencies.
- `WRFV3/dyn_em/module_initialize_ideal.F` - updated model initiation (particularly for `quarter_ss` ideal case):
	- set `mminlu2` to `USGS` and water land-use code to 16.
	- set all surface area in the model to water at prescribed sea-surface temperature (SST).
	- set eta-levels to be read from namelist.
	- fixed bug in hydrostatic rebalancing of ph_1 where c1h(k) and c2h(k) were used instead of c1h(k-1) and c2h(k-1) as in the first calculation of ph_1.
	- set variable `TMN` (soil minimum temperature) to `TSK-0.5` (K) where `TSK` is SST.
- `WRFV3/dyn_em/module_nudging.F` - new module containing the following function:
	- `apply_light_nudging`: nudge variables towards the grid average of the variable (without using a tendency variable).
- `WRFV3/dyn_em/solve_em.F`:
	- added printout of information.
	- added light nudging code into third runge-kutta step. 
	- updated call to `phy_prep_part2` to pass new tendencies.
- `WRFV3/dyn_em/start_em.F` - updated call to `phy_init` to pass new tendencies.
- `WRFV3/phys/module_physics_addtendc.F` - updated function `update_phy_tend` to accept new tendencies and add them to tendency sums.
- `WRFV3/phys/module_physics_init.F` - updated function `phy_init` to accept new tendencies and initialise them to zero.
- `WRFV3/phys/module_ra_rrtmg_sw.F` - updated to fix solar constant as per Yi-Ling Hwong's code.
- `WRFV3/phys/module_radiation_driver.F` - updated to fix solar constant as per Yi-Ling Hwong's code.
- `WRFV3/phys/module_sf_sfclayrev.F` - implemented ideal evaporation and setting of surface wind, as per Yi-Ling Hwong's code.
	- updated function `sfclayrev` to accept ideal evaporation and surface wind options.
	- updated function `sfclayrev1d` and call it it, to accept/pass ideal evaporation and surface wind options.
	- if ideal evaporation is used, surface moist and surface heat fluxes are set to be constant according to the value of the surface wind.
- `WRFV3/phys/module_surface_driver.F`:
	- updated function `surface_driver` to accept ideal evaporation and surface wind options.
	- updated calls to and functions of `sfclayrev_seaice_wrapper` to pass/accept ideal evaporation and surface wind options.
	- updated call to `sfclayrev` to pass ideal evaporation and surface wind options.
- `WRFV3/share/output_wrf.F`
	- updated output to include the values of important options in netcdf files.
