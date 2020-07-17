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

## Modifications to WRF

### General notes

* The radiative cooling profile is prescribed when `const_rad_cooling == 1`. In this case the radiation driver is called, and immediately afterwards the theta tendancy due to radiation is prescribed. Note this may affect the accuracy of the following variables which are computed by the radiation driver: `COSZEN, CLDFRA, SWDOWN, GLW, ACSWUPT, ACSWUPTC, ACSWDNT, ACSWDNTC, ACSWUPB, ACSWUPBC, ACSWDNB, ACSWDNBC, ACLWUPT, ACLWUPTC, ACLWUPB, ACLWUPBC, ACLWDNB, ACLWDNBC, SWUPT, SWUPTC, SWDNT, SWDNTC, SWUPB, SWUPBC, SWDNB, SWDNBC, LWUPT, LWUPTC, LWUPB, LWUPBC, LWDNB, LWDNBC, OLR`.

### Specific changes

Note that theta means potential temperature (K), QV means water vapour mixing ratio (kg kg-1), U and V are horizontal winds (m s-1).

Changes are made to the following files:

- `WRFV3/Registry/Registry.EM_COMMON`
	- Made `RTHRATEN` (potential temperature tendency due to radiation scheme) an output variable.
	- Added new grid variables:
		- `RTHFORCETEN`, `RQVFORCETEN` - tendency due to perturbation forcing in U and V respectively.
		- `RUFORCETEN`, `RVFORCETEN` - tendency due to wind relaxation in U and V respectively.
		- `RELAX_U_TARGET_PROFILE`, `RELAX_V_TARGET_PROFILE` - the target U and V wind profiles for wind relaxation.
	- Added control options: 
		- `ssttsk` - sea surface temperature for initial state (K).
		- `use_light_nudging` - use light nudging for wind fields?
		- `ideal_evap_flag` - use ideal evaporation?
		- `surface_wind` - surface wind value for initial state (m s-1).
		- `const_rad_cooling` - use prescribed, constant radiative cooling profile?
		- `relax_uv_winds` - relax U and V winds to target profiles?
		- `wind_relaxation_time` - wind relaxation time (s).
		- `relax_u_profile_file` - filename for target U wind profile.
		- `relax_v_profile_file` - filename for target V wind profile.
		- `perturb_t` - perturb temperature (theta)?
		- `perturb_q` - perturb moisture (QV)?  
		- `k_pert` - the 1-based vertical level index to apply perturbations to.
		- `TtendAmp` - amplitude of theta perturbation (K day-1).
		- `QtendAmp` - amplitude of QV perturbation (kg kg-1 day-1).

```
WRFV3/dyn_em:
  Makefile                         - 
  module_LRF.F                     -
  module_big_step_utilities_em.F   - 
  module_diffusion_em.F            -
  module_em.F                      -
  module_first_rk_step_part1.F     -
  module_first_rk_step_part2.F     - 
  module_initialize_ideal.F        - fixed bug in hydrostatic rebalancing of ph_1 where c1h(k) and c2h(k)
				     were used instead of c1h(k-1) and c2h(k-1) as in the first
				     calculation of ph_1.
  module_nudging.F                 - 
  solve_em.F                       -
  start_em.F                       -

WRFV3/phys:
  module_physics_addtendc.F        -
  module_physics_init.F            - 
  module_ra_rrtmg_sw.F             - 
  module_radiation_driver.F        -
  module_sf_sfclayrev.F            -
  module_surface_driver.F          -

WRFV3/share:
  output_wrf.F                     -
```
