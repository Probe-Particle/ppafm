
! took from here 
! Smith, Oliver K. (April 1961), "Eigenvalues of a symmetric 3 × 3 matrix.", Communications of the ACM 4 (4): 168
! http://www.geometrictools.com/Documentation/EigenSymmetric3x3.pdf
! and wiki http://en.wikipedia.org/wiki/Eigenvalue_algorithm

subroutine eignum3x3 ( Ain, eignums )
 implicit none
! == Local Parameters and Data Declaration
	real, dimension (3,3), intent(in)	:: Ain
	real, dimension (3),   intent(out)	:: eignums
	real, parameter :: inv3  = 0.33333333333D0 ! 1/3
	real, parameter :: root3 = 1.73205080757D0 ! sqrt(3)
! == Local variables
	real, dimension (3,3)				:: A
	real amax, c0, c1, c2, c2Div3, aDiv3, mbDiv2, magnitude, angle, cs, sn, aij, q
	integer i,j
! == Procedure
	amax = 0
	amax = -1e-300
	do i = 1,3
		!write (*,'(3f20.10)') Ain(:,i)
		do j = 1,3
			aij = Ain(j,i)
			amax = max( amax, aij  ) 
		end do
		!write (*,*)
	end do
	!write (*,*) amax
	A(:,:)=Ain(:,:)/amax
	c0 = A(1,1)*A(2,2)*A(3,3) + 2*A(1,2)*A(1,3)*A(2,3) - a(1,1)*a(2,3)*a(2,3) - A(2,2)*A(1,3)*A(1,3) - A(3,3)*a(1,2)*a(1,2)
	c1 = A(1,1)*A(2,2) - A(1,2)*A(1,2) + A(1,1)*A(3,3) - A(1,3)*A(1,3) + A(2,2)*A(3,3) - A(2,3)*A(2,3)
	c2 = A(1,1) + A(2,2) + A(3,3)
	!write (*,'(A,3f20.10)') "c1,c0,c2: ",c0,c1,c2
	c2Div3 = c2*inv3
	aDiv3 = (c1 - c2*c2Div3)*inv3
	if (aDiv3 > 0.0) aDiv3 = 0.0
	mbDiv2 = 0.5*( c0 + c2Div3*(2.0*c2Div3*c2Div3 - c1) )
	q = mbDiv2*mbDiv2 + aDiv3*aDiv3*aDiv3
	if (q > 0.0) q = 0.0
	magnitude = sqrt(-aDiv3)
	angle = atan2( sqrt(-q),mbDiv2) *inv3
	cs = cos(angle);
	sn = sin(angle);
	!write (*,'(A,4f20.10)') "magnitude,angle,cs,sn: ",magnitude,angle,cs,sn
	eignums(1) = amax*( c2Div3 + 2.0*magnitude*cs          )
	eignums(2) = amax*( c2Div3 - magnitude*(cs + root3*sn) )
	eignums(3) = amax*( c2Div3 - magnitude*(cs - root3*sn) )
end subroutine eignum3x3

