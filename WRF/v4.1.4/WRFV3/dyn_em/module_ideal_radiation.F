module module_ideal_radiation

! AUTHOR: Yi-Ling Hwong (CCRC); updated by T. Raupach (CCRC).
! Implements constant radiative cooling profile as per Herman and Kuang (2013).
! To implement Herman & Kuang (2013) Linear Response Function (LRF).

CONTAINS
  
  subroutine fix_radiation( &
       p_hyd, pi_phy, rthraten,      &
       ids, ide, jds, jde, kds, kde, &
       ims, ime, jms, jme, kms, kme, &
       ips, ipe, jps, jpe, kps, kpe, &
       its, ite, jts, jte, kts, kte)

    ! Set the radiation profile to a constant cooling profile as per Herman and Kuang (2013).
    !
    ! p_hyd: hydrostatic pressure at mass points.
    ! pi_phy: the Exner function (absolute temperature/potential temperature).
    ! rthraten: theta tendency due to radiation; set by this function.
    ! ids, ide, jds, jde: kds, kde: Horizontal domain dimensions.
    ! ims, ime, jms, jme, kms, kme: Memory dimensions.
    ! ips, ipe, jps, jpe, kps, kpe: Patch dimensions.
    ! its, ite, jts, jte, kts, kte: Tile dimensions.
    
    implicit none
    
    real, dimension(ims:ime,kms:kme,jms:jme), intent(in)    :: p_hyd, pi_phy
    real, dimension(ims:ime,kms:kme,jms:jme), intent(inout) :: rthraten
    integer, intent(in)                                     :: &
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
end module module_ideal_radiation
