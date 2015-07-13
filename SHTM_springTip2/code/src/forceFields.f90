


subroutine getFF_LJ  ( R,  Z, from, E, F)
    use G_globals
 implicit none
! == parameters
  real, dimension (3), intent(in)    :: R
  integer            , intent(in)    :: Z
  type (subSystem)   , intent(in)    :: from
  real,                intent(inout) :: E
  real, dimension (3), intent(inout) :: F
! == variables
        integer i
        real, dimension (3) :: dR
		real FF 
		real ir2,ir6,ir12
! == body
do i = 1, from%n
  dR(:) = from%Rs(:,i) - R(:) 
  ir2  = 1.0/dot_product( dR, dR ) 
  ! ir2  = 1.0/ ( dR(1)**2 + dR(2)**2 + dR(3)**2 ) 
  ir6  = ir2**3
  ir12 = ir6**2
  E = E +  (      c12ij( Z, from%Zs(i))*ir12 -    c6ij( Z, from%Zs(i))*ir6  )
  FF   =   (  -12*c12ij( Z, from%Zs(i))*ir12 +  6*c6ij( Z, from%Zs(i))*ir6  ) * ir2
  F(:) = F(:) + FF*dR(:)
end do ! i
!write (*,'(A,4f10.5)') " in getFF_LJ ", E, F(:)
end subroutine getFF_LJ






subroutine getFF_vdW  ( R,  Z, from, E, F)
    use G_globals
 implicit none
! == parameters
  real, dimension (3), intent(in)    :: R
  integer            , intent(in)    :: Z
  type (subSystem)   , intent(in)    :: from
  real,                intent(inout) :: E
  real, dimension (3), intent(inout) :: F
! == variables
        integer i
        real, dimension (3) :: dR
		real FF 
		real ir2,ir6,ir12
! == body
do i = 1, from%n
  dR(:) = from%Rs(:,i) - R(:) 
  ir2  = 1.0/dot_product( dR, dR ) 
  ! ir2  = 1.0/ ( dR(1)**2 + dR(2)**2 + dR(3)**2 ) 
  ir6  = ir2**3
  ir12 = ir6**2
  E = E +     c6ij( Z, from%Zs(i))*ir6
  FF   =    6*c6ij( Z, from%Zs(i))*ir6 * ir2
  F(:) = F(:) + FF*dR(:)
end do ! i
!write (*,'(A,4f10.5)') " in getFF_LJ ", E, F(:)
end subroutine getFF_vdW




subroutine getFF_Pauli  ( R,  Z, from, E, F)
    use G_globals
 implicit none
! == parameters
  real, dimension (3), intent(in)    :: R
  integer            , intent(in)    :: Z
  type (subSystem)   , intent(in)    :: from
  real,                intent(inout) :: E
  real, dimension (3), intent(inout) :: F
! == variables
        integer i
        real, dimension (3) :: dR
		real FF 
		real ir2,ir6,ir12
! == body
do i = 1, from%n
  dR(:) = from%Rs(:,i) - R(:) 
  ir2  = 1.0/dot_product( dR, dR ) 
  ! ir2  = 1.0/ ( dR(1)**2 + dR(2)**2 + dR(3)**2 ) 
  ir6  = ir2**3
  ir12 = ir6**2
  E = E +  (      c12ij( Z, from%Zs(i))*ir12  )
  FF   =   (  -12*c12ij( Z, from%Zs(i))*ir12  ) * ir2
  F(:) = F(:) + FF*dR(:)
end do ! i
!write (*,'(A,4f10.5)') " in getFF_LJ ", E, F(:)
end subroutine getFF_Pauli









subroutine getFF_Morse ( R, Z, from, E, F)
    use G_globals
 implicit none
! == parameters
  real, dimension (3), intent(in)  :: R
  integer, intent(in)    :: Z
  type (subSystem)   , intent(in)  :: from
  real,                intent(inout) :: E
  real, dimension (3), intent(inout) :: F
! == variables
        integer i
        real, dimension (3) :: dR
		real E0, a, rr, expar 
! == body
do i = 1, from%n
 dR(:) = from%Rs(:,i) - R(:) 
 E0    = sqrt( atypes(Z)%E0 * atypes(from%Zs(i))%E0  )
 a     = sqrt( kMorse/ E0                       )
 rr    = sqrt( dR(1)**2 + dR(2)**2 + dR(3)**2    )
 expar = exp ( a* ( atypes(Z)%R0 + atypes(from%Zs(i))%R0  - rr ) )
 E     = E    +          E0 * (    expar*expar - 2*expar )
 F(:)  = F(:) - dR(:)* ( E0 * (  2*expar*expar - 2*expar ) * -a )  
! Write (*,'(A,4f16.8)') "E0,a,rr,expar",E0,a,rr, expar
end do ! i
end subroutine getFF_Morse



subroutine getFF_Harmonic ( R, E, F)
    use G_globals
 implicit none
! == parameters
  real, dimension (3), intent(in)    :: R
  real,                intent(inout) :: E
  real, dimension (3), intent(inout) :: F
! == variables
  real, dimension (3) :: FF
! == body
  FF(:) = kHarmonic(:) * R(:)
  E    = E    + dot_product( FF, R )
  F(:) = F(:) - FF(:)
end subroutine getFF_Harmonic



subroutine getFF_Rspring ( R, E, F)
    use G_globals
 implicit none
! == parameters
  real, dimension (3), intent(in)    :: R
  real,                intent(inout) :: E
  real, dimension (3), intent(inout) :: F
! == variables
  real l, dl
! == body
  l      = sqrt( dot_product( R, R ) )
  dl   = l - l0_radial
  F(:) = F(:) -  R(:) * ( k_radial * dl / l )
  E    = E    +  0.5 * k_radial * dl * dl 
end subroutine getFF_Rspring





subroutine getFF_HarmonicTip ( Rprobe, E, F)
    use G_globals
 implicit none
! == parameters
  real, dimension (3), intent(in)    :: Rprobe
  real,                intent(inout) :: E
  real, dimension (3), intent(inout) :: F
! == variables
  real, dimension (3) :: FF
! == body
  FF(:) = kHarmonic(:) * ( Rprobe(:) - Rtip(:) - RspringMin(:) - Rprobe0(:) )
  E    = E    + dot_product( FF, Rprobe )
  F(:) = F(:) - FF(:)
end subroutine getFF_HarmonicTip





! according to  http://cacs.usc.edu/education/phys516/04TB.pdf
! Tight-Binding Model of Electronic Structures
subroutine getHoppingPP ( beta, R, a1, a2,  from, T )
    use G_globals
 implicit none
! == Local Parameters and Data Declaration
  real,                intent(in)    :: beta
  real, dimension (3), intent(in)    :: R
  real, dimension (3), intent(in)    :: a1
  real, dimension (3), intent(in)    :: a2
  type (subSystem)   , intent(in)    :: from
  real,                intent(inout) :: T
! == Local Variable Declaration and Description
        integer i
        real, dimension (3) :: dR, d
		real rr, h, c1,c2, pi,sigma
! == Procedure
T = 0
do i = 1, from%n
	if (  from%Zs(i) .gt. 1 ) then
		dR(:) = from%Rs(:,i) - R(:)                     
		rr    = sqrt( dR(1)**2 + dR(2)**2 + dR(3)**2  )
		d     = dR / rr
		h     = exp( -beta*rr )                  ! R-dependence
		c1    = dot_product(a1,d) 
		c2    = dot_product(a2,d)
		sigma = c1*c2*h
		pi    = dot_product( a1 - c1*d, a2-c2*d )*h
		T = T + (sigma - pi)**2
	end if ! atypes( from%Zs(i) )
end do ! i
end subroutine getHoppingPP


subroutine getHoppingSS ( beta, R, from, T )
    use G_globals
 implicit none
! == Local Parameters and Data Declaration
  real,                intent(in)    :: beta
  real, dimension (3), intent(in)    :: R
  type (subSystem)   , intent(in)    :: from
  real,                intent(inout) :: T
! == Local Variable Declaration and Description
        integer i
        real, dimension (3) :: dR
		real rr, h
! == Procedure
T = 0
do i = 1, from%n
	if (  from%Zs(i) .gt. 1 ) then
		dR(:) = from%Rs(:,i) - R(:)                     
		rr    = sqrt( dR(1)**2 + dR(2)**2 + dR(3)**2  )
		h     = exp( -beta*rr )
		T = T + h
	end if ! atypes( from%Zs(i) )
end do ! i
end subroutine getHoppingSS


