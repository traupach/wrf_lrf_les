! AUTHOR: Yi-Ling Hwong (CCRC), with modifications by Tim Raupach, 2020.
!
! Functions for forcing vertical temperature or moisture tendencies to implement
! Herman & Kuang (2013) Linear Response Function (LRF).

module module_LRF
  
CONTAINS

  subroutine force_LRF( &
       perturb_t, perturb_q, & 
       p_pert, TtendAmp, QtendAmp, &
       p, pi, thten, qvten, &
       ids, ide, jds, jde, kds, kde, &
       ims, ime, jms, jme, kms, kme, &
       ips, ipe, jps, jpe, kps, kpe, &
       its, ite, jts, jte, kts, kte)
    ! force_LRF: force vertical temperature and moisture tendencies, to
    ! implement Herman & Kuang (2013) Linear Response Function (LRF).
    !
    ! NOTE: Tendencies thten and qvten are overwritten by this function.
    !
    ! ---
    ! perturb_t: Perturb temperature tendencies?
    ! perturb_q: Perturb moisture tendencies?
    ! p_pert: The pressure around which to perturb [hPa].
    ! TtendAmp: Temperature (dT/dt) perturbation [K day-1].
    ! QtendAmp: Water vapour mixing ratio (dq/dt) perturbation [kg kg-1 day-1].
    ! p: Pressure field [Pa].
    ! pi: The Exner function (pi_phy in WRF).
    ! thten: Potential temperature (theta) tendency field [K s-1].
    ! qvten: Water vapour tendency field [kg kg-1 s-1].
    ! ids, ide, jds, jde: kds, kde: Horizontal domain dimensions.
    ! ims, ime, jms, jme, kms, kme: Memory dimensions.
    ! ips, ipe, jps, jpe, kps, kpe: Patch dimensions.
    ! its, ite, jts, jte, kts, kte: Tile dimensions.
    
    implicit none

    ! Input variables.
    logical, intent(in)                                  :: perturb_t, perturb_q
    real, intent(in)                                     :: p_pert
    real, intent(in)                                     :: TtendAmp, QtendAmp
    real, dimension(ims:ime,kms:kme,jms:jme), intent(in) :: p, pi
    integer, intent(in)                                  :: ids,ide,jds,jde,kds,kde, &
                                                            ims,ime,jms,jme,kms,kme, &  
                                                            ips,ipe,jps,jpe,kps,kpe, &
                                                            its,ite,jts,jte,kts,kte  

    ! Input/output variables.
    real, dimension(ims:ime,kms:kme,jms:jme), intent(inout) :: thten ! dT/dt (K s-1).
    real, dimension(ims:ime,kms:kme,jms:jme), intent(inout) :: qvten ! dq/dt (kg kg-1 s-1).
    
    ! Local variables.
    integer                                  :: i,j,k
    real                                     :: forcing
    

    ! If neither temperature or moisture are to be perturbed, return.
    if ((perturb_t .eqv. .false.) .and. (perturb_q .eqv. .false.)) return
  
    ! The forcing form is from the one used in Appendix A of Kuang 2012. Their
    ! forcing is around a vertical level and is the mean of a delta function and
    ! gamma function. We adapr the form to use only use the gamma function part
    ! and perturb around a pressure, not a level. This is because the pressure
    ! levels in the model may not exactly align with the pressure around which
    ! we want to perturb.

    ! Loop over staggered grid.
    do i = its, min(ite, ide-1)
       do k = kts, min(kte, kde-1)
          do j = jts, min(jte, jde-1)

             forcing = exp(-((p(i,k,j)-(p_pert*100))/7500.)**2.)
             
             ! Temperature forcing as a forcing on potential temperature (theta)
             ! tendency. Convert dT/dt to theta tendency by dividing by the
             ! exner function.
             if(perturb_t .eqv. .true.) then
                thten(i,k,j) = ((TtendAmp/86400) * forcing) / pi(i,k,j)
             endif
             
             ! Moisture forcing as a forcing on moisture tendency.
             if(perturb_q .eqv. .true.) then
                qvten(i,k,j) = (QtendAmp/86400) * forcing
             endif
          enddo
       enddo
    enddo
  end subroutine force_LRF

  subroutine relax_winds_to_profile( &
       uten, vten, u, v, tau,        &
       target_wind_u, target_wind_v, &
       ids, ide, jds, jde, kds, kde, &
       ims, ime, jms, jme, kms, kme, &
       ips, ipe, jps, jpe, kps, kpe, &
       its, ite, jts, jte, kts, kte)
    ! Relax U and V winds towards profiles read from file.
    !
    ! NOTE: uten and vten are overwritten by this function.
    ! 
    ! ---
    ! uten, vten: The U and V wind tendency fields [m s-2].
    ! u, v: The U and V wind fields (at mass points) [m s-1].
    ! tau: Relaxation time, over which a full relaxation is made [s].
    ! target_wind_u: The target profile for U [m s-1].
    ! target_wind_v: The target profile for V [m s-1].
    ! ids, ide, jds, jde: kds, kde: Horizontal domain dimensions.
    ! ims, ime, jms, jme, kms, kme: Memory dimensions.
    ! ips, ipe, jps, jpe, kps, kpe: Patch dimensions.
    ! its, ite, jts, jte, kts, kte: Tile dimensions.

    implicit none
 
    real, dimension(ims:ime, kms:kme, jms:jme), intent(in)    :: u, v
    real, dimension(ims:ime, kms:kme, jms:jme), intent(inout) :: uten, vten
    real, intent(in)                 :: tau
    real, dimension(kte), intent(in) :: target_wind_u, target_wind_v
    integer, intent(in) :: &
         ids, ide, jds, jde, kds, kde, &
         ims, ime, jms, jme, kms, kme, &
         ips, ipe, jps, jpe, kps, kpe, &
         its, ite, jts, jte, kts, kte

    integer :: i,j,k

    ! Relax U and V winds. 
    do j=jts,min(jte, jde-1)
       do k=kts,min(kte, kde-1)
          do i=its,min(ite, ide-1)
             uten(i,k,j) = (1/tau) * -(u(i,k,j) - target_wind_u(k))
             vten(i,k,j) = (1/tau) * -(v(i,k,j) - target_wind_v(k))
          enddo
       enddo
    enddo
  end subroutine relax_winds_to_profile

  subroutine relax_stratospheric_t_qv(     &
       t_phy, qv, thten, qvten, p, pi, &
       target_t, target_q,                 &
       ids, ide, jds, jde, kds, kde,       &
       ims, ime, jms, jme, kms, kme,       &
       ips, ipe, jps, jpe, kps, kpe,       &
       its, ite, jts, jte, kts, kte)
    ! Relax T and q in the stratosphere to match a given target profile, as per
    ! Figure 1 in Herman and Kuang (2013).
    !
    ! NOTE: tendencies thten and qvten are added to by this function.
    !
    ! ---
    ! t_phy: Temperature (theta) field [K].
    ! qv: Water vapour mixing ratio field [kg kg-1].
    ! thten: Potential temperature (theta) tendency field [K s-1].
    ! qvten: Water vapour tendency field [kg kg-1 s-1].
    ! p: Hydrostatic pressure [Pa].
    ! pi: The Exner function (pi_phy in WRF).
    ! target_t: Target profile for temperature T [K].
    ! target_q: Target profile for q [kg kg-1].
    ! ids, ide, jds, jde: kds, kde: Horizontal domain dimensions.
    ! ims, ime, jms, jme, kms, kme: Memory dimensions.
    ! ips, ipe, jps, jpe, kps, kpe: Patch dimensions.
    ! its, ite, jts, jte, kts, kte: Tile dimensions.

    implicit none

    real, dimension(ims:ime,kms:kme,jms:jme), intent(inout) :: t_phy, qv
    real, dimension(ims:ime,kms:kme,jms:jme), intent(inout) :: thten ! dT/dt (K s-1).
    real, dimension(ims:ime,kms:kme,jms:jme), intent(inout) :: qvten ! dq/dt (kg kg-1 s-1)
    real, dimension(ims:ime,kms:kme,jms:jme), intent(in)    :: p
    real, dimension(ims:ime,kms:kme,jms:jme), intent(in)    :: pi
    real, dimension(kte), intent(in) :: target_t, target_q
    integer, intent(in) :: &
         ids, ide, jds, jde, kds, kde, &
         ims, ime, jms, jme, kms, kme, &
         ips, ipe, jps, jpe, kps, kpe, &
         its, ite, jts, jte, kts, kte
    
    integer :: i, j, k
    real    :: inv_tau, min_tau, relax_t, relax_q

    ! The minimum value of tau is two days in seconds, meaning the maximum
    ! inverse relaxation constant (1/tau = inv_tau) is 0.5 day-1 as per Herman
    ! and Kuang 2013.
    min_tau = 2*24*3600

    write(6,*) 'k pressure inv_tau'
    
    do i = its, min(ite, ide-1)
       do k = kts, min(kte, kde-1)
          do j = jts, min(jte, jde-1)
             if (p(i,k,j) .ge. 16000) then
                ! Below 160 hPa, don't do any relaxation of T or q. 
                relax_t = 0.0
                relax_q = 0.0
             else
                if (p(i,k,j) .le. 10000) then
                   ! Above 100 hPa, use the full relaxation time.
                   inv_tau = 1/min_tau
                else
                   ! Between 160 hPa and 100 hPa, vary 1/tau linearly from 0 day-1
                   ! at 160 hPa to 0.5 day-1 at 100 hPa.
                   inv_tau = 0.5*(16000 - p(i,k,j))/6000

                   ! Convert from day-1 to s-1.
                   inv_tau = inv_tau/(24*3600)
                endif

                ! Relaxation amount per timestep.
                relax_t = inv_tau * -(t_phy(i,k,j) - target_t(k))
                relax_q = inv_tau * -(qv(i,k,j) - target_q(k))
             endif

             if(i .eq. 1 .and. j .eq. 1) then
                write(6,*) k, p(i,k,j), inv_tau
             endif
             
             ! Relax the variables by the relaxation amount. Convert temperature
             ! to potential temperature (theta) for tendency.
             thten(i,k,j) = thten(i,k,j) + (relax_t / pi(i,k,j)) 
             qvten(i,k,j) = qvten(i,k,j) + relax_q
          enddo
       enddo
    enddo
  end subroutine relax_stratospheric_t_qv
  
  subroutine read_profile_file(num_levels, profile_file, vals)
    ! Read a profile from file. Expect one value per line.
    ! ---
    ! num_levels: The number of levels (values) to read.
    ! profile_file: The file to read from.
    ! vals: The read values (output).

    implicit none
    integer, intent(in) :: num_levels
    character(len=*), intent(in) :: profile_file
    real, dimension(num_levels), intent(inout) :: vals

    integer :: k
    
    open(unit=10, file=profile_file, form='formatted', status='old')
    rewind(10)

    k = 1
    do while(k .lt. num_levels)
       read(10,*) vals(k)
       k = k + 1
    enddo

    close(unit=10, status='keep')
  end subroutine read_profile_file
  
  subroutine read_wind_target_profiles(num_levels, u_profile_file, &
       v_profile_file, u_val, v_val)
    ! Read U and V profiles from two files.
    ! ---
    ! num_levels: The number of levels (values) to read, one per line.
    ! u_profile_file: The file to read from for U.
    ! v_profile_file: The file to read from for V.
    ! u_val: The read values (output) for U.
    ! v_val: The read values (output) for V.
    
    implicit none
    integer, intent(in) :: num_levels
    character(len=*), intent(in) :: u_profile_file, v_profile_file
    real, dimension(num_levels), intent(inout) :: u_val, v_val
    write(6,*) 'Loading target profiles for wind relaxation.'
    write(6,*) 'U target profile file:', u_profile_file
    write(6,*) 'V target profile file:', v_profile_file
    
    call read_profile_file(num_levels, u_profile_file, u_val)
    call read_profile_file(num_levels, v_profile_file, v_val)
  end subroutine read_wind_target_profiles

  subroutine read_tq_target_profiles(num_levels, t_profile_file, &
       q_profile_file, t_val, q_val)
    ! Read U and V profiles from two files.
    ! ---
    ! num_levels: The number of levels (values) to read, one per line.
    ! t_profile_file: The file to read from for T.
    ! q_profile_file: The file to read from for q.
    ! t_val: The read values (output) for T.
    ! q_val: The read values (output) for q.
    
    implicit none
    integer, intent(in) :: num_levels
    character(len=*), intent(in) :: t_profile_file, q_profile_file
    real, dimension(num_levels), intent(inout) :: t_val, q_val
    write(6,*) 'Loading target profiles for T and q stratospheric relaxation.'
    write(6,*) 'T target profile file:', t_profile_file
    write(6,*) 'q target profile file:', q_profile_file
    
    call read_profile_file(num_levels, t_profile_file, t_val)
    call read_profile_file(num_levels, q_profile_file, q_val)
  end subroutine read_tq_target_profiles

  subroutine fix_radiation(          &
       p, pi, rthraten,              &
       ids, ide, jds, jde, kds, kde, &
       ims, ime, jms, jme, kms, kme, &
       ips, ipe, jps, jpe, kps, kpe, &
       its, ite, jts, jte, kts, kte)

    ! Set the radiation profile to a constant cooling profile as per Herman and
    ! Kuang (2013).
    !
    ! p: hydrostatic pressure at mass points (Pa).
    ! pi: the Exner function (absolute temperature/potential temperature).
    ! rthraten: theta tendency due to radiation; set by this function.
    ! ids, ide, jds, jde: kds, kde: Horizontal domain dimensions.
    ! ims, ime, jms, jme, kms, kme: Memory dimensions.
    ! ips, ipe, jps, jpe, kps, kpe: Patch dimensions.
    ! its, ite, jts, jte, kts, kte: Tile dimensions.
    
    implicit none
    
    real, dimension(ims:ime,kms:kme,jms:jme), intent(in) :: p, pi
    real, dimension(ims:ime,kms:kme,jms:jme), intent(inout) :: rthraten
    integer, intent(in) :: &
         ids, ide, jds, jde, kds, kde, &
         ims, ime, jms, jme, kms, kme, &
         ips, ipe, jps, jpe, kps, kpe, &
         its, ite, jts, jte, kts, kte
    
    integer :: i,j,k
    real :: t_tend
    
    do j = jts, min(jte, jde-1)
       do k = kts, min(kte, kde-1)
          do i = its, min(ite, ide-1)

             if (p(i,k,j) .ge. 20000) then             
                ! From surface to 200 hPa, constant radiative cooling value
                ! of Qrad = -1.5 K day-1 (converted to K s-1).
                t_tend = -1.5/86400
             else if (p(i,k,j) .le. 10000) then
                ! Above 100 hPa, radiative cooling is zero.
                t_tend = 0.0
             else
                ! Linear change between 200 hPa and 100 hPa.
                t_tend = -((1.5*p(i,k,j)/10000 - 1.5) / 86400)   
             endif
             
             ! Convert from temperature tendency to theta tendency.
             rthraten(i,k,j) = t_tend/pi(i,k,j) 
          enddo
       enddo
    enddo
  end subroutine fix_radiation
  
end module module_LRF
