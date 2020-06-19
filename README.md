# WRF modifications for LRF perturbations

Implementation in WRF LES of Kuang (2010)'s linear response function.

Changes are made to the following files:

```
WRFV3/Registry:
  Registry.EM_COMMON

WRFV3/dyn_em:
  Makefile
  module_LRF.F
  module_big_step_utilities_em.F
  module_diffusion_em.F
  module_em.F
  module_first_rk_step_part1.F
  module_first_rk_step_part2.F
  module_initialize_ideal.F
  module_nudging.F
  solve_em.F
  start_em.F

WRFV3/phys:
  module_physics_addtendc.F
  module_physics_init.F
  module_ra_rrtmg_sw.F
  module_radiation_driver.F
  module_sf_sfclayrev.F
  module_surface_driver.F

WRFV3/share:
  output_wrf.F
```