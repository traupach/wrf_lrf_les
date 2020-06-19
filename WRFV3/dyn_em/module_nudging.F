!WRF:MEDIATION_LAYER:SOLVER
  
! Nudging routines for WRF variables. Adapted code, sources are:
!     - light nudging routine: David Fuchs and Maxime Colin.
! Adaption and commenting by Tim Raupach, 2020.

#define BENCH_START(A)
#define BENCH_END(A)

MODULE module_nudging
  
CONTAINS
  
  SUBROUTINE apply_light_nudging ( &
       varname, var_2, dt_model, nudging_time, &
       stagger_x, stagger_y, stagger_z, print_info, &
       ids, ide, jds, jde, kds, kde, &
       ims, ime, jms, jme, kms, kme, &
       its, ite, jts, jte, kts, kte)
    ! apply_light_nudging: nudge all values of var_2 towards the grid average
    ! value of var_2 for each level.
    ! ---
    ! varname: Name of the variable to nudge.
    ! var_2: The variable to nudge.
    ! dt_model: The grid time step (dt).
    ! nudging_time: The nudging time scale [s]. 
    ! stagger_x, stagger_y, stagger_z: Staggered in x/y/z dimensions?
    ! print_info: print debugging information?
    ! ids, ide, jds, jde: kds, kde: Horizontal domain dimensions.
    ! ims, ime, jms, jme, kms, kme: Memory dimensions.
    ! its, ite, jts, jte, kts, kte: Tile dimensions.
    
    USE module_dm, ONLY: wrf_dm_sum_real
    
    IMPLICIT NONE

    CHARACTER(LEN=*), INTENT(IN) :: varname
    REAL, DIMENSION(ims:ime, kms:kme, jms:jme), INTENT(INOUT) :: var_2
    
    INTEGER, INTENT(IN) :: &
         ids, ide, jds, jde, kds, kde, &
         ims, ime, jms, jme, kms, kme, &
         its, ite, jts, jte, kts, kte

    REAL, INTENT(IN) :: dt_model
    REAL, INTENT(IN) :: nudging_time
    LOGICAL, INTENT(IN) :: stagger_x, stagger_y, stagger_z
    LOGICAL, INTENT(IN) :: print_info
    
    REAL :: nudging_coef
    REAL :: sum_var_hor              
    REAL :: count_hor                
    REAL, DIMENSION(kts:kte) :: var_hor_average
    INTEGER :: i, j, k, i_from, i_to, j_from, j_to, k_from, k_to, stag_x, stag_y
    
    ! Define the nudging coefficient using time scale.
    nudging_coef = dt_model / nudging_time
    
    ! Determine tile indices based on whether variable is staggered or not. Assume unstaggered to
    ! start with:
    i_from = max(its, ids)
    i_to = min(ite, ide-1)
    j_from = max(jts, jds)
    j_to = min(jte, jde-1)
    k_from = max(kts, kds)
    k_to = min(kte, kde-1)
    
    ! Some variables are staggered; when the variable is staggered it has one more point on the
    ! staggered dimension (to account for the final boundary).
    stag_x = 0
    stag_y = 0
    if(stagger_x) then
       i_to = min(ite, ide)
       stag_x = 1
    end if

    if(stagger_y) then
       j_to = min(jte, jde)
       stag_y = 1
    end if

    if(stagger_z) then
       k_to = min(kte, kde)
    end if
    
    ! Loop through vertical levels in the tile.
    DO k = k_from, k_to
       ! Find the sum of var_2 and count the number of points in the tile.
       sum_var_hor = 0.
       count_hor = 0.
       DO j = j_from, j_to
          DO i = i_from, i_to
             sum_var_hor = sum_var_hor + var_2(i,k,j)
             count_hor = count_hor + 1.
          END DO
       END DO

       ! Use MPI to compute find global sum and total number of points over all tiles.
       sum_var_hor = wrf_dm_sum_real(sum_var_hor)
       count_hor = wrf_dm_sum_real(count_hor)

       ! Check that all domain points have been counted.
       if(.not. count_hor .eq. (ide-ids+stag_x) * (jde-jds+stag_y)) then
          write(6,*) 'Light nudging for', varname, ': ERROR: horizontal mean does', &
               ' not include all domain points.'
          write(6,*) 'Points in average: count_hor = ', count_hor
          write(6,*) 'Points in domain: (ide-ids+stag_x) * (jde-jds+stag_y) = ', &
               (ide-ids+stag_x) * (jde-jds+stag_y)
          call wrf_error_fatal("Light nudging grid mean error.")
       endif

       ! Find the global horizontal average for var_2.
       var_hor_average(k) = sum_var_hor / count_hor
       
       ! Loop through grid points at this level and nudge by subtracting horizontal average, such
       ! that in nudging_time seconds the entire average would be subtracted.
       DO j = j_from, j_to
          DO i = i_from, i_to
             var_2(i,k,j) = var_2(i,k,j) - var_hor_average(k) * nudging_coef
          END DO
       END DO
    END DO

    if(print_info .eqv. .true.) then
       write(6,*) 'Loop extents i:', i_from, i_to
       write(6,*) 'Loop extents j:', j_from, j_to
       write(6,*) 'Loop extents k:', k_from, k_to
       write(6,*) '---'       
       write(6,*) 'Light nudging for variable:', varname
       write(6,*) 'Grid average first level (k_from):', var_hor_average(k_from)
       write(6,*) 'Grid average last level (k_to):', var_hor_average(k_to)
       write(6,*) 'Points in each average: ', count_hor
       write(6,*) '---'
       write(6,*) 'Dimensions in this thread:'
       write(6,*) 'ids,ide,jds,jde:', ids, ide, jds, jde
       write(6,*) 'ims,ime,jms,jme:', ims, ime, jms, jme
       write(6,*) 'its,ite,jts,jte:', its, ite, jts, jte
       write(6,*) 'kds,kde:', kds, kde
       write(6,*) 'kms,kme:', kms, kme
       write(6,*) 'kts,kte:', kts, kte
       write(6,*) '---'
       write(6,*) 'Shape of var_average is', shape(var_hor_average)
       write(6,*) 'Shape of var_2 is', shape(var_2)
       write(6,*) '---'
       write(6,*) 'nudging coefficient is', nudging_coef
    end if
  END SUBROUTINE apply_light_nudging  
  
END MODULE module_nudging
