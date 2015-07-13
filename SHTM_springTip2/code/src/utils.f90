

subroutine cross3d(a,b,c)
implicit none
	real, dimension (3), intent(in)   :: a,b  
	real, dimension (3), intent(out)  :: c 
	c(1)=a(2)*b(3)-a(3)*b(2)
	c(2)=a(3)*b(1)-a(1)*b(3)
	c(3)=a(1)*b(2)-a(2)*b(1)
end subroutine cross3d

subroutine inv3D ( A, Ainv )
      implicit none
      real, dimension (3,3), intent(in)  :: A
      real, dimension (3,3), intent(out) :: Ainv
      real :: idet
      idet =  1/( A(1,1)*A(2,2)*A(3,3) - A(1,1)*A(2,3)*A(3,2) - A(1,2)*A(2,1)*A(3,3) + A(1,2)*A(2,3)*A(3,1) + A(1,3)*A(2,1)*A(3,2) - A(1,3)*A(2,2)*A(3,1) )
      Ainv(1,1) = +( A(2,2)*A(3,3)-A(2,3)*A(3,2) ) * idet
      Ainv(2,1) = -( A(2,1)*A(3,3)-A(2,3)*A(3,1) ) * idet
      Ainv(3,1) = +( A(2,1)*A(3,2)-A(2,2)*A(3,1) ) * idet
      Ainv(1,2) = -( A(1,2)*A(3,3)-A(1,3)*A(3,2) ) * idet
      Ainv(2,2) = +( A(1,1)*A(3,3)-A(1,3)*A(3,1) ) * idet
      Ainv(3,2) = -( A(1,1)*A(3,2)-A(1,2)*A(3,1) ) * idet
      Ainv(1,3) = +( A(1,2)*A(2,3)-A(1,3)*A(2,2) ) * idet
      Ainv(2,3) = -( A(1,1)*A(2,3)-A(1,3)*A(2,1) ) * idet
      Ainv(3,3) = +( A(1,1)*A(2,2)-A(1,2)*A(2,1) ) * idet
end subroutine inv3D 
