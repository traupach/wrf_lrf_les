# WRF modifications for LRF perturbations

Installation:

```
cd <WRFDIR>
git clone -b V4.1.4 git@github.com:coecms/WRF.git
cd <GITDIR>/wrf_lrf_les/WRF/
code/sh/install_wrf.sh <WRFDIR>/WRF/
```

Compilation:

```
cd <WRFDIR>/WRF/WRFV3/
./run_compile --clean --compile_case em_quarter_ss
```

Implementation in WRF LES of Kuang (2010)'s linear response function.

* The radiative cooling profile is prescribed when `const_rad_cooling == 1`. In this case the radiation driver is called, and immediately afterwards the theta tendancy due to radiation is prescribed. Note this may affect the accuracy of the following variables which are computed by the radiation driver: `COSZEN, CLDFRA, SWDOWN, GLW, ACSWUPT, ACSWUPTC, ACSWDNT, ACSWDNTC, ACSWUPB, ACSWUPBC, ACSWDNB, ACSWDNBC, ACLWUPT, ACLWUPTC, ACLWUPB, ACLWUPBC, ACLWDNB, ACLWDNBC, SWUPT, SWUPTC, SWDNT, SWDNTC, SWUPB, SWUPBC, SWDNB, SWDNBC, LWUPT, LWUPTC, LWUPB, LWUPBC, LWDNB, LWDNBC, OLR`.

Changes are made to the following files:

```
WRFV3/Registry:
  Registry.EM_COMMON               - made RTHRATEN an output variable.
  				   - Added new variables RTHFORCETEN, RQVFORCETEN,
				     RUFORCETEN, RVFORCETEN, RELAX_U_TARGET_PROFILE,
				     and RELAX_V_TARGET_PROFILE.
				   - Added control options ssttsk, use_light_nudging,
				     ideal_evap_flag, surface_wind, const_rad_cooling,
				     relax_uv_winds, wind_relaxation_time,
				     relax_u_profile_file, relax_v_profile_file,
				     perturb_t, perturb_q, k_pert, TtendAmp, and
				     QtendAmp.

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
