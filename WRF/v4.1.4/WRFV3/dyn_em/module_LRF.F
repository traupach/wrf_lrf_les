! AUTHOR: Yi-Ling Hwong (CCRC), with modifications by Tim Raupach, 2020.
!
! Forces vertical temperature or moisture tendencies to implement Herman & Kuang
! (2013) Linear Response Function (LRF).

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

    integer                                    :: i,j,k
    real, dimension(kte)                       :: target_wind_u, target_wind_v
    character (len=256)                        :: message       
    real                                       :: tau

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

  subroutine read_profile_file(num_levels, profile_file, vals)
    ! Read a profile from file.
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
  
  subroutine read_target_profiles(num_levels, u_profile_file, v_profile_file, &
       u_val, v_val)
    ! Read U and V profiles from two files.
    ! ---
    ! num_levels: The number of levels (values) to read.
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

  end subroutine read_target_profiles

end module module_LRF
