

subroutine move_simple( R, v, F)
	use G_globals
 implicit none
! == Local Parameters and Data Declaration
	real, dimension (3), intent(inout)  :: R
	real, dimension (3), intent(inout)  :: v
	real, dimension (3), intent(in)     :: F
! == Procedure
	v = v + F*dt
	v = v*(1.0-damping)
	R = R + v*dt
end subroutine move_simple

! =================== FIRE pptimization

subroutine move_FIRE_1( R, v, F)
	use G_globals
 implicit none
! == Local Parameters and Data Declaration
	real, dimension (3), intent(inout)  :: R
	real, dimension (3), intent(inout)  :: v
	real, dimension (3), intent(in)     :: F
! == Local variables
	real, dimension (3)   :: vhat, Fhat
	real                  :: vlen
! == Procedure
	Fhat = F / ( sqrt(dot_product(F,F)) + 0.1e-12 )
	vlen = ( sqrt(dot_product(v,v)) + 0.1e-12 )
	vhat = v / vlen
	v = v + F*dt
	v = v - damping*vlen*( vhat - Fhat )
	R = R + v*dt
	! write (*,'(A,10f10.5)') " dt,v,F,Fhat", dt,v,F,Fhat
end subroutine move_FIRE_1


! =================== FIRE pptimization

subroutine initFIRE( )
	use G_globals
 implicit none
   FIRE_finc    = 1.1 
   FIRE_fdec    = 0.5
   FIRE_falpha  = 0.99
   FIRE_Nmin    = 5
   FIRE_dtmax  = dt 
   FIRE_dt     = dt 
   FIRE_acoef0 = damping
   FIRE_acoef  = FIRE_acoef0 
end subroutine initFIRE


subroutine move_FIRE_true( R, v, F)
	use G_globals
 implicit none
! == Local Parameters and Data Declaration
	real, dimension (3), intent(inout)  :: R
	real, dimension (3), intent(inout)  :: v
	real, dimension (3), intent(in)     :: F
! == Local variables
	real  :: ff,vv,vf, cF, cV
! == Procedure
    ff = dot_product(F,F)
    vv = dot_product(v,v)
    vf = dot_product(v,F)
	if ( vf .lt. 0 ) then
      v(:)       = 0 
      FIRE_dt    = FIRE_dt * FIRE_fdec
      FIRE_acoef = FIRE_acoef0
    else
      cF           =    FIRE_acoef * sqrt(vv/ff)
      cV          = 1 - FIRE_acoef
      v(:)        = cV * v(:)  + cF * F(:)
      FIRE_dt     = min( FIRE_dt * FIRE_finc, FIRE_dtmax ) 
      FIRE_acoef  = FIRE_acoef * FIRE_falpha
    end if
	v = v + F * FIRE_dt
	R = R + v * FIRE_dt
end subroutine move_FIRE_true


! ===============================
! ==================  relax    - Relaxation using pairwise LJ + elecrostatic grid


subroutine relax( Rprobe, Fprobe, Eprobe  )
	use G_globals
 implicit none
! == Local Parameters and Data Declaration
!  	type (subSystem)   , intent(in)     :: system
!	real, dimension (3), intent(in)     :: Rtip
	real, dimension (3), intent(inout)  :: Rprobe
	real, dimension (3), intent(out)    :: Fprobe
	real, intent(out)                   :: Eprobe
! == Local variables
	integer i
	real, dimension (3)   :: v
	real                  :: Etemp
! == Procedure
	if (relaxMethod .eq. 3) call initFIRE()
	if (startKick .gt. 0) then
        relaxItersSum = relaxItersSum + 1
		Eprobe = 0
		Fprobe(:) = 0.D0
		call FFelec%interpolate ( Rprobe, Fprobe                       ) ! Electrostatic Force from surface
		call getFF_LJ          ( Rprobe, probeZ,  surf, Eprobe, Fprobe ) ! Surface potential
		call getFF_LJ          ( Rprobe, probeZ,   tip, Eprobe, Fprobe  ) ! Tip poteitnal
		call getFF_Rspring     ( Rprobe - Rtip, Eprobe,        Fprobe )
		call getFF_HarmonicTip ( Rprobe, Eprobe, Fprobe                )
		! RTipprobe = Rprobe - Rtip - RspringMin
		! call getFF_Harmonic    ( RTipprobe,  Eprobe, Fprobe)  ! Tip poteitnal
		v = startKick*dt*Fprobe
	else
		v(:) = 0
	end if
	do iiter = 1, maxRelaxIter
        relaxItersSum = relaxItersSum + 1
		Eprobe = 0
		Fprobe(:) = 0.D0
		call FFelec%interpolate ( Rprobe,                       Fprobe )  ! Electrostatic Force from surface
		call getFF_LJ          ( Rprobe, probeZ,    surf, Eprobe, Fprobe )  ! Surface potential
		!call getFF_LJ          ( Rprobe, probeZ,    tip, Eprobe, Fprobe )  ! Tip poteitnal
		call getFF_Rspring     ( Rprobe - Rtip,  Eprobe,        Fprobe )
		call getFF_HarmonicTip ( Rprobe, Eprobe,                Fprobe )
		! RTipprobe = Rprobe - Rtip - RspringMin
		! call getFF_Harmonic ( RTipprobe,  Eprobe, Fprobe)      ! Tip poteitnal
		select case (relaxMethod)
    		case (1)
				call move_simple(Rprobe,v,Fprobe)
    		case (2)
				call move_FIRE_1(Rprobe,v,Fprobe)
            case (3)
				call move_FIRE_true(Rprobe,v,Fprobe)
 		end select
		if ( wrtDebug .gt. 1 ) write (*,'(A,i5,f25.5,5x, 3f25.5,5x, 3f25.5 )') 'i,E,Fx,Fy,Fz,Rx,Ry,Rz ',i, Eprobe, Fprobe, Rprobe
		if (dot_product(Fprobe,Fprobe) .lt. convergF**2) exit
		if (wrtDebug .gt. 2) then
			write (2000,'(i5)')   (1+tip%n + surf%n) 
			write (2000,'(A,i5,3f20.10)') "  iz, iteration ", iiter, Rtip
			write (2000, '( A, 3f20.8 )') atypes(probeZ)%symbol, Rprobe
			call system2XYZ ( tip, 2000)
			call system2XYZ ( surf, 2000)
		end if ! wrtDebug 
	end do
	Fprobe(:) = 0.D0
    call FFelec%interpolate ( Rprobe, Fprobe )                        ! Electrostatic Force from surface
	call getFF_LJ          ( Rprobe, probeZ, surf, Etemp, Fprobe )  ! Surface potential
	if (iiter .ge. maxRelaxIter ) then
		write (*,*) " Not converged in ",maxRelaxIter,' steps '
	end if
end  subroutine relax


! ===============================
! ==================  relaxGrid  - Relaxation using FFgrid(3,ix,iy,iz) for surface potential


subroutine relaxGrid( Rprobe, Fprobe, Eprobe  )
	use G_globals
 implicit none
! == Local Parameters and Data Declaration
!  	type (subSystem)   , intent(in)     :: system
!	real, dimension (3), intent(in)     :: Rtip
	real, dimension (3), intent(inout)  :: Rprobe
	real, dimension (3), intent(out)    :: Fprobe
	real, intent(out)                   :: Eprobe
! == Local variables
	integer i
	real, dimension (3)   :: v
!	real                  :: Etemp
! == Procedure
	if (relaxMethod .eq. 3) call initFIRE()
	if (startKick .gt. 0) then
        relaxItersSum = relaxItersSum + 1
		Eprobe = 0
		Fprobe(:) = 0.D0
		call FFgrid%interpolate ( Rprobe,                      Fprobe  ) ! surface ForceField on grid              
		! call getFF_LJ           ( Rprobe, probeZ,   tip, Eprobe, Fprobe  ) ! Tip poteitnal
		call getFF_Rspring     ( Rprobe - Rtip, Eprobe,        Fprobe )
		call getFF_HarmonicTip  ( Rprobe, Eprobe, Fprobe                )
		! RTipprobe = Rprobe - Rtip - RspringMin - Rprobe0
		! call getFF_Harmonic    ( RTipprobe,  Eprobe, Fprobe)  ! Tip poteitnal
		v = startKick*dt*Fprobe
	else
		v(:) = 0
	end if
	do iiter = 1, maxRelaxIter
        relaxItersSum = relaxItersSum + 1
		Eprobe = 0
		Fprobe(:) = 0.D0
		call FFgrid%interpolate ( Rprobe,                       Fprobe )  ! surface ForceField on grid      
		!call getFF_LJ           ( Rprobe, probeZ,  tip, Eprobe, Fprobe )  ! Tip poteitnal
		call getFF_Rspring      ( Rprobe - Rtip, Eprobe,        Fprobe )
		call getFF_HarmonicTip  ( Rprobe, Eprobe,               Fprobe )
		! RTipprobe = Rprobe - Rtip - RspringMin
		! call getFF_Harmonic ( RTipprobe,  Eprobe, Fprobe )      ! Tip poteitnal
		select case (relaxMethod)
    		case (1)
				call move_simple(Rprobe,v,Fprobe)
    		case (2)
				call move_FIRE_1(Rprobe,v,Fprobe)
            case (3)
				call move_FIRE_true(Rprobe,v,Fprobe)
 		end select
		if ( wrtDebug .gt. 1 ) write (*,'(A,i5,f25.5,5x, 3f25.5,5x, 3f25.5 )') 'i,E,Fx,Fy,Fz,Rx,Ry,Rz ',i, Eprobe, Fprobe, Rprobe
		if (dot_product(Fprobe,Fprobe) .lt. convergF**2) exit
		if (wrtDebug .gt. 2) then
			write (2000,'(i5)')   (1+tip%n + surf%n) 
			write (2000,'(A,i5,3f20.10)') "  iz, iteration ", iiter, Rtip
			write (2000, '( A, 3f20.8 )') atypes(probeZ)%symbol, Rprobe
			call system2XYZ ( tip, 2000)
			call system2XYZ ( surf, 2000)
		end if ! wrtDebug 
	end do
	Fprobe(:) = 0.D0
	call FFgrid%interpolate ( Rprobe, Fprobe )                       ! surface ForceField on grid   
	if (iiter .ge. maxRelaxIter ) then
		write (*,*) " relaxGrid Not converged in ",maxRelaxIter,' steps '
		! Fprobe(:) = 10000.D0
		! write (*,*) "DEBUG 0 Fprobe =",Fprobe(:)
	end if
end  subroutine relaxGrid













