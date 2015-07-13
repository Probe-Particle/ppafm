

! ===============================
! ==================  Dynamical Matrix Pairwise Atoms

subroutine dynmat( Rprobe, eignums  )
	use G_globals
implicit none
! == Local Parameters and Data Declaration
!  	type (subSystem)   , intent(in)     :: system
	! real, dimension (3), intent(in)     :: Rtip
	real, dimension (3), intent(in)     :: Rprobe
	real, dimension (3), intent(out)    :: eignums
! == Local variables
	integer i
	real, dimension (3)   :: f1,f2, Rdisp    !  v
	real, dimension (3,3) :: D
	real                  :: tmp
! == Procedure
	do i = 1,3
		! === displace in positive direction
		Rdisp(:) = Rprobe(:)
		Rdisp(i) = Rdisp(i) + ddisp 
		f1(:) = 0.D0
		call FFelec%interpolate ( Rdisp,                     f1 ) 
		call getFF_LJ          ( Rdisp, probeZ, surf, tmp, f1 )  ! Surface potential
		call getFF_LJ          ( Rdisp, probeZ,  tip, tmp, f1 )  ! Tip poteitnal
		call getFF_HarmonicTip ( Rdisp,  tmp,               f1 ) 
		! RTipprobe = Rdisp - Rtip - RspringMin
		! call getFF_Harmonic ( RTipprobe,  tmp, f1 )             ! Tip poteitnal
		! === displace in negative direction
		Rdisp(:) = Rprobe(:)
		Rdisp(i) = Rdisp(i) - ddisp 
		f2(:) = 0.D0
		call FFelec%interpolate ( Rdisp,                     f2 ) 
		call getFF_LJ          ( Rdisp, probeZ, surf, tmp, f2 )  ! Surface potential
		call getFF_LJ          ( Rdisp, probeZ,  tip, tmp, f2 )  ! Tip poteitnal
		call getFF_HarmonicTip ( Rdisp,  tmp,               f2 ) 
		!RTipprobe = Rdisp - Rtip - RspringMin
		!call getFF_Harmonic ( RTipprobe,  tmp, f2 )             ! Tip poteitnal
		! === build dinamical matrix
		D(i,:) = ( f2 - f1 )/(2*ddisp)
	end do

	! symetrize 
	tmp    = ( D(1,2) + D(2,1) )*0.5
	D(1,2) = tmp
	D(2,1) = tmp
	tmp    = ( D(1,3) + D(3,1) )*0.5
	D(1,3) = tmp
	D(3,1) = tmp
	tmp    = ( D(2,3) + D(3,2) )*0.5
	D(2,3) = tmp
	D(3,2) = tmp
	
	! find eigen-numbers
	call eignum3x3( D, eignums ) 

	! ordering from lowest to highest energy
	if( eignums(3) .lt. eignums(1)  ) then 
		tmp = eignums(1) 
		eignums(1) = eignums(3)
		eignums(3) = tmp
	end if
	if( eignums(2) .lt. eignums(1)  ) then 
		tmp = eignums(1) 
		eignums(1) = eignums(2)
		eignums(2) = tmp
	end if
	if( eignums(3) .lt. eignums(2)  ) then 
		tmp = eignums(2) 
		eignums(2) = eignums(3)
		eignums(3) = tmp
	end if

end  subroutine dynmat


! ===============================
! ==================  Dynamical Matrix On Grid

subroutine dynmatGrid( Rprobe, eignums  )
	use G_globals
implicit none
! == Local Parameters and Data Declaration
!  	type (subSystem)   , intent(in)     :: system
	! real, dimension (3), intent(in)     :: Rtip
	real, dimension (3), intent(in)     :: Rprobe
	real, dimension (3), intent(out)    :: eignums
! == Local variables
	integer i
	real, dimension (3)   :: f1,f2, Rdisp   ! v
	real, dimension (3,3) :: D
	real                  :: tmp
! == Procedure
!	write (*,*) "Dynmat DEBUG 1"
	do i = 1,3
		! === displace in positive direction
		Rdisp(:) = Rprobe(:)
		Rdisp(i) = Rdisp(i) + ddisp 
		f1(:) = 0.D0
		call FFgrid%interpolate ( Rdisp,                     f1 ) 
		call getFF_LJ           ( Rdisp, probeZ,    tip, tmp, f1 )  ! Tip poteitnal
		call getFF_HarmonicTip  ( Rdisp,  tmp,               f1 ) 
		! RTipprobe = Rdisp - Rtip - RspringMin
		! call getFF_Harmonic ( RTipprobe,  tmp, f1 )              ! Tip poteitnal
		! === displace in negative direction
		Rdisp(:) = Rprobe(:)
		Rdisp(i) = Rdisp(i) - ddisp 
		f2(:) = 0.D0
		call FFgrid%interpolate ( Rdisp,                     f2 ) 
		call getFF_LJ           ( Rdisp, probeZ,    tip, tmp, f2 )  ! Tip poteitnal
		call getFF_HarmonicTip  ( Rdisp,  tmp,               f2 ) 
		!RTipprobe = Rdisp - Rtip - RspringMin
		!call getFF_Harmonic ( RTipprobe,  tmp, f2 )               ! Tip poteitnal
		! === build dinamical matrix
		D(i,:) = ( f2 - f1 )/(2*ddisp)
	end do

!	write (*,*) "Dynmat DEBUG 2"
	! symetrize 
	tmp    = ( D(1,2) + D(2,1) )*0.5
	D(1,2) = tmp
	D(2,1) = tmp
	tmp    = ( D(1,3) + D(3,1) )*0.5
	D(1,3) = tmp
	D(3,1) = tmp
	tmp    = ( D(2,3) + D(3,2) )*0.5
	D(2,3) = tmp
	D(3,2) = tmp

!	write (*,*) "Dynmat DEBUG 3"	
	! find eigen-numbers
	call eignum3x3( D, eignums ) 

!	write (*,*) "Dynmat DEBUG 4"
	! ordering from lowest to highest energy
	if( eignums(3) .lt. eignums(1)  ) then 
		tmp = eignums(1) 
		eignums(1) = eignums(3)
		eignums(3) = tmp
	end if
	if( eignums(2) .lt. eignums(1)  ) then 
		tmp = eignums(1) 
		eignums(1) = eignums(2)
		eignums(2) = tmp
	end if
	if( eignums(3) .lt. eignums(2)  ) then 
		tmp = eignums(2) 
		eignums(2) = eignums(3)
		eignums(3) = tmp
	end if

end  subroutine dynmatGrid
