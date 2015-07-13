
program testEigen
  implicit none
	integer i
	real, dimension (3,3) :: A
	real, dimension (3)   :: eignums
! code

	! A = reshape((/ 1, 2, 3, 4, 5, 6, 7, 8, 9 /), shape(A))
	A = reshape((/ 1.0,-3.0,-2.0,-3.0,2.0,-1.0,-2.0,-1.0,3.0 /), shape(A))
	call eignum3x3( A, eignums )

	write (*,'(3f20.10)') eignums

	!do i = 1,3
	!		write (*,*,ad) A(:,i)
	!	write (*,*)
	!end do

end program testEigen

