! AUTHOR: Yi-Ling Hwong (CCRC), with modifications by Tim Raupach, 2020.
!
! Functions for forcing vertical temperature or moisture tendencies to implement
! Herman & Kuang (2013) Linear Response Function (LRF).

module module_LRF
  
CONTAINS
  
  subroutine force_LRF( &
       perturb_t, perturb_q, & 
       k_pert, TtendAmp, QtendAmp, &
       p, pi, thten, qvten, &
       ids, ide, jds, jde, kds, kde, &
       ims, ime, jms, jme, kms, kme, &
       ips, ipe, jps, jpe, kps, kpe, &
       its, ite, jts, jte, kts, kte)
    ! force_LRF: force vertical temperature and moisture tendencies, to
    ! implement Herman & Kuang (2013) Linear Response Function (LRF).
    ! ---
    ! perturb_t: Perturb temperature tendencies?
    ! perturb_q: Perturb moisture tendencies?
    ! k_pert: The index of the level (k) to perturb.
    ! TtendAmp: Temperature (dT/dt) perturbation [K day-1].
    ! QtendAmp: Water vapour mixing ratio (dq/dt) perturbation [kg kg-1 day-1].
    ! p: Pressure field.
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
    integer, intent(in)                                  :: k_pert
    real, intent(in)                                     :: TtendAmp, QtendAmp
    real, dimension(ims:ime,kms:kme,jms:jme), intent(in) :: p, pi
    integer, intent(in)                                  :: ids,ide,jds,jde,kds,kde, &
                                                            ims,ime,jms,jme,kms,kme, &  
                                                            ips,ipe,jps,jpe,kps,kpe, &
                                                            its,ite,jts,jte,kts,kte  

    ! Input/output variables.
    real, dimension(ims:ime,kms:kme,jms:jme), intent(inout) :: thten 
    real, dimension(ims:ime,kms:kme,jms:jme), intent(inout) :: qvten 
    
    ! Local variables.
    real, dimension(ims:ime,kms:kme,jms:jme) :: tten   ! dT/dt (K/s).
    real, dimension(ims:ime,kms:kme,jms:jme) :: qten   ! dq/dt (kg/kg/s).
    real                                     :: deltaf ! Dirac delta function.
    integer                                  :: i,j,k


    ! If neither temperature or moisture are to be perturbed, return.
    if ((perturb_t .eqv. .false.) .and. (perturb_q .eqv. .false.)) return
  
    ! The forcing form is from the one used in Appendix A of Kuang 2012. The
    ! difference btw this form and the one used in Equation (4) of Herman and
    ! Kuang 2013 is a factor of 1/2. Note 86400 is number of s in a day.

    ! Loop over staggered grid.
    do i = its, min(ite, ide-1)
       do k = kts, min(kte, kde-1)
          do j = jts, min(jte, jde-1)

             ! Delta function.
             if(k == k_pert) then
                deltaf = 1.0
             else
                deltaf = 0.0
             endif
             
             ! Temperature forcing as a forcing on potential temperature
             ! (theta) tendency.
             if(perturb_t .eqv. .true.) then
                tten(i,k,j) = 0.5 * (TtendAmp/86400 * & 
                     (deltaf + exp(-((p(i,k,j)-p(i,k_pert,j))/7500.)**2.)))

                ! Convert dT/dt to theta tendency.
                thten(i,k,j) = tten(i,k,j) / pi(i,k,j)
             endif
             
             ! Moisture forcing as a forcing on moisture tendency.
             if(perturb_q .eqv. .true.) then
                qten(i,k,j) = 0.5 * (QtendAmp/86400 * & 
                     (deltaf + exp(-((p(i,k,j)-p(i,k_pert,j))/7500.)**2.)))
                
                qvten(i,k,j) = qten(i,k,j)
             endif
          enddo
       enddo
    enddo
  end subroutine force_LRF

  subroutine relax_winds_to_profile(uten, vten, u, v, &
       dt, relaxation_time, &
       target_wind_u, target_wind_v, &
       ids, ide, jds, jde, kds, kde, &
       ims, ime, jms, jme, kms, kme, &
       ips, ipe, jps, jpe, kps, kpe, &
       its, ite, jts, jte, kts, kte)
    ! Relax U and V winds towards profiles read from file. 
    ! ---
    ! uten, vten: The U and V wind tendency fields [m s-1].
    ! u, v: The U and V wind fields [m s-1].
    ! relaxation_time: The time over which a full relaxation would be made [s].
    ! dt: Timestep length [s].
    ! target_wind_u: The target profile for U [m s-1].
    ! target_wind_v: The target profile for V [m s-1].
    ! ids, ide, jds, jde: kds, kde: Horizontal domain dimensions.
    ! ims, ime, jms, jme, kms, kme: Memory dimensions.
    ! ips, ipe, jps, jpe, kps, kpe: Patch dimensions.
    ! its, ite, jts, jte, kts, kte: Tile dimensions.

    implicit none
 
    real, intent(in)                                          :: relaxation_time
    real, intent(in)                                          :: dt
    real, dimension(ims:ime, kms:kme, jms:jme), intent(in)    :: u,v
    real, dimension(ims:ime, kms:kme, jms:jme), intent(inout) :: uten,vten
    integer, intent(in) :: &
         ids, ide, jds, jde, kds, kde, &
         ims, ime, jms, jme, kms, kme, &
         ips, ipe, jps, jpe, kps, kpe, &
         its, ite, jts, jte, kts, kte

    integer              :: i,j,k
    real, dimension(kte) :: target_wind_u, target_wind_v
    real                 :: tau

    ! Determine how many time steps there are in the relaxation time.
    tau = relaxation_time / dt

    ! Relax U and V winds. 
    do j=jts,min(jte, jde-1)
       do k=kts,min(kte, kde-1)
          do i=its,min(ite, ide-1)
             uten(i,k,j) = -(u(i,k,j) - target_wind_u(k)) / tau
             vten(i,k,j) = -(v(i,k,j) - target_wind_v(k)) / tau
          enddo
       enddo
    enddo
  end subroutine relax_winds_to_profile

  subroutine relax_stratospheric_t_qv( &
       t_phy, qv, p_hyd, dt,           &
       relax_t, relax_q,               &
       target_t, target_q,             &
       ids, ide, jds, jde, kds, kde,   &
       ims, ime, jms, jme, kms, kme,   &
       ips, ipe, jps, jpe, kps, kpe,   &
       its, ite, jts, jte, kts, kte)
    ! Relax T and q in the stratosphere to match a given target profile, as per
    ! Figure 1 in Herman and Kuang (2013).
    !
    ! t_phy: The theta field to adjust [K].
    ! qv: The water vapour mixing ratio field to adjust [kg kg-1].
    ! p_hyd: Hydrostatic pressure [Pa].
    ! dt: Timestep length [s].
    ! relax_t: Output variable to store the relaxation amount for T [K].
    ! relax_q: Output variable to store the relaxation amount of q [kg kg-1].
    ! target_t: Target profile for T [K].
    ! target_q: Target profile for q [kg kg-1].
    ! ids, ide, jds, jde: kds, kde: Horizontal domain dimensions.
    ! ims, ime, jms, jme, kms, kme: Memory dimensions.
    ! ips, ipe, jps, jpe, kps, kpe: Patch dimensions.
    ! its, ite, jts, jte, kts, kte: Tile dimensions.

    implicit none
    
    real, dimension(ims:ime,kms:kme,jms:jme), intent(inout) :: t_phy, qv
    real, dimension(ims:ime,kms:kme,jms:jme), intent(in)    :: p_hyd
    real, dimension(ims:ime,kms:kme,jms:jme), intent(out)   :: relax_t, relax_q
    real, dimension(kte), intent(in) :: target_t, target_q
    real, intent(in) :: dt
    integer, intent(in) :: &
         ids, ide, jds, jde, kds, kde, &
         ims, ime, jms, jme, kms, kme, &
         ips, ipe, jps, jpe, kps, kpe, &
         its, ite, jts, jte, kts, kte
    
    integer :: i, j, k
    real    :: tau, relaxation_time

    ! Two days in seconds, meaning the maximum relaxation constant is 0.5 day-1
    ! as per Herman and Kuang 2013.
    relaxation_time = 2*24*3600 
    
    do i = its, min(ite, ide-1)
       do k = kts, min(kte, kde-1)
          do j = jts, min(jte, jde-1)
             if (p_hyd(i,k,j) .ge. 16000) then
                ! Below 160 hPa, don't do any relaxation of T or q. 
                relax_t(i,k,j) = 0.0
                relax_q(i,k,j) = 0.0
             else
                if (p_hyd(i,k,j) .le. 10000) then
                   ! Above 100 hPa, use the full relaxation time.
                   tau = relaxation_time
                else
                   ! Between 160 hPa and 100 hPa, vary tau linearly.
                   tau = ((16000 - p_hyd(i,k,j)) * relaxation_time) / 10000
                endif

                ! Convert tau from per second to per timestep.
                tau = tau / dt
                
                ! Relaxation amount per timestep.
                relax_t(i,k,j) = -(t_phy(i,k,j) - target_t(k)) / tau
                relax_q(i,k,j) = -(qv(i,k,j) - target_q(k)) / tau
             endif

             ! Relax the variables by the relaxation amoujnt.
             t_phy(i,k,j) = t_phy(i,k,j) + relax_t(i,k,j)
             qv(i,k,j) = qv(i,k,j) + relax_q(i,k,j)
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

  subroutine fix_radiation( &
       p_hyd, pi_phy, rthraten,      &
       ids, ide, jds, jde, kds, kde, &
       ims, ime, jms, jme, kms, kme, &
       ips, ipe, jps, jpe, kps, kpe, &
       its, ite, jts, jte, kts, kte)

    ! Set the radiation profile to a constant cooling profile as per Herman and
    ! Kuang (2013).
    !
    ! p_hyd: hydrostatic pressure at mass points.
    ! pi_phy: the Exner function (absolute temperature/potential temperature).
    ! rthraten: theta tendency due to radiation; set by this function.
    ! ids, ide, jds, jde: kds, kde: Horizontal domain dimensions.
    ! ims, ime, jms, jme, kms, kme: Memory dimensions.
    ! ips, ipe, jps, jpe, kps, kpe: Patch dimensions.
    ! its, ite, jts, jte, kts, kte: Tile dimensions.
    
    implicit none
    
    real, dimension(ims:ime,kms:kme,jms:jme), intent(in) :: p_hyd, pi_phy
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

             if (p_hyd(i,k,j) .ge. 20000) then             
                ! From surface to 200 hPa, constant radiative cooling value
                ! of Qrad = -1.5 K day-1 (converted to K s-1).
                t_tend = -1.5/86400
             else if (p_hyd(i,k,j) .le. 10000) then
                ! Above 100 hPa, radiative cooling is zero.
                t_tend = 0.0
             else
                ! Linear change between 200 hPa and 100 hPa.
                t_tend = -((1.5*p_hyd(i,k,j)/10000 - 1.5) / 86400)   
             endif
             
             ! Convert from temperature tendency to theta tendency.
             rthraten(i,k,j) = t_tend/pi_phy(i,k,j) 
          enddo
       enddo
    enddo
  end subroutine fix_radiation
  
end module module_LRF
