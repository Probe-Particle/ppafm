module T_grid3Dvec3
	use T_grid3D	

	implicit none


type grid3Dvec3                                         ! 3D grid maps of observables
  real,    dimension (3) :: Rmin, Rmax, Rspan, step, invStep ! grid size and spacing (x,y,z)
  integer, dimension (3) :: N                       ! dimension in x,y,z
  real, dimension    (:,:, :, :), allocatable :: f     ! array storing values of othe observable
  contains
    procedure :: initgrid    =>  grid3Dvec3_initgrid 
    procedure :: copySetup   =>  grid3Dvec3_copySetup
    procedure :: echoSetup   =>  grid3Dvec3_echoSetup
	!procedure :: fromGrids   =>  grid3Dvec3_fromGrids
    !procedure :: toGrids     =>  grid3Dvec3_toGrids
    procedure :: writeXSF    =>  grid3Dvec3_writeXSF
    procedure :: fromXSF     =>  grid3Dvec3_fromXSF
	procedure :: interpolate =>  grid3Dvec3_interpolate
end type grid3Dvec3

contains 

! ===== grid3Dvec3 :: initgrid
 subroutine grid3Dvec3_initgrid ( this )
 implicit none
  ! === variables
  class (grid3Dvec3) :: this
  ! === body
  this%Rspan(:) = this%Rmax(:) - this%Rmin(:)
  this%N(1) = floor ( this%Rspan(1)/ this%step(1) ) + 1
  this%N(2) = floor ( this%Rspan(2)/ this%step(2) ) + 1
  this%N(3) = floor ( this%Rspan(3)/ this%step(3) ) + 1
  this%invStep(:) = 1/this%step(:)
  allocate (this%f( 3, this%N(1),this%N(2),this%N(3) ))
 end subroutine grid3Dvec3_initgrid

! ===== grid3Dvec3 :: setByTemplate
subroutine grid3Dvec3_copySetup ( this, from )
 implicit none
  ! === variables
  class (grid3Dvec3) :: this
  class (grid3Dvec3) :: from
  ! === body
  this%N(:)       = from%N(:)
  this%step(:)    = from%step(:)
  this%invStep(:) = from%invStep(:)
  this%Rmin(:)    = from%Rmin(:)
  this%Rmax(:)    = from%Rmax(:)
  this%Rspan(:)   = from%Rspan(:)
  allocate (this%f( 3, this%N(1),this%N(2),this%N(3) ))
end subroutine grid3Dvec3_copySetup


! ===== grid3Dvec3 :: fromGrids
!subroutine grid3Dvec3_fromGrids ( this, from_x, from_y, from_z )
! implicit none
!  ! === variables
!  class (grid3Dvec3) :: this
!  class (grid3D    ) :: from_x, from_y, from_z
!  integer i,j,k
!  ! === body
!  this%N(:) = from_x%N(:)
!  allocate (this%f( 3, this%N(1),this%N(2),this%N(3) ))
!  this%Rmin(:)  = from_x%Rmin(:)
!  this%Rmax(:)  = from_x%Rmax(:)
!  this%step(:)  = from_x%step(:)
!  this%invStep(:) = 1/this%step(:)
!  this%Rspan(:) = from_x%Rspan(:)
!  call this%echoSetup()
!  do k=1, this%N(3)  ! layer 
!    do j=1,  this%N(2)
!       do i=1,  this%N(1)
!         this%f(1,i,j,k) = from_x%f(i,j,k) 
!         this%f(2,i,j,k) = from_y%f(i,j,k) 
!         this%f(3,i,j,k) = from_z%f(i,j,k) 
!       end do !i
!    end do !j
! end do !k
!end subroutine grid3Dvec3_fromGrids

! ===== grid3Dvec3 :: toGrids 
!subroutine grid3Dvec3_toGrids ( this, from_x, from_y, from_z )
! implicit none
!  ! variables
!  class (grid3Dvec3) :: this
!  class (grid3D    ) :: from_x, from_y, from_z
!  integer i,j,k
!  ! body
!  do k=1, this%N(3)  ! layer 
!    do j=1,  this%N(2)
!       do i=1,  this%N(1)
!         from_x%f(i,j,k) = this%f(1,i,j,k) 
!         from_y%f(i,j,k) = this%f(2,i,j,k) 
!         from_z%f(i,j,k) = this%f(3,i,j,k) 
!       end do !i
!    end do !j
! end do !k
!end subroutine grid3Dvec3_toGrids

! ===== grid3Dvec :: echoSetup 
subroutine grid3Dvec3_echoSetup ( this )
 implicit none
   ! variables
   class (grid3Dvec3) :: this
   ! body
   write (*,'(A,3f16.8)') "Rmin(:)    ",this%Rmin(:)
   write (*,'(A,3f16.8)') "Rmax(:)    ",this%Rmax(:)
   write (*,'(A,3f16.8)') "step(:)    ",this%step(:)
   write (*,'(A,3f16.8)') "invStep(:) ",this%invStep(:)
   write (*,'(A,3f16.8)') "Rspan(:)   ",this%Rspan(:)
   write (*,'(A,3i6   )') "N(:)       ",this%N(:)
end subroutine grid3Dvec3_echoSetup

! ===== grid3D :: fromXSF
subroutine grid3Dvec3_fromXSF(this, prename )
 implicit none
  ! === parameters
 class (grid3Dvec3) :: this
 character (*) prename 
 ! === variables
 character (40) fname
 character(1)  fnum
 integer :: i, j, k
 integer iline, ifile
 character (100) line 
 real,    dimension (3,4) :: lvs 
 ! === body
 do ifile = 1,3
     write (fnum,'(i1.1)') ifile
     fname = prename//"_"//fnum//'.xsf'
	 write (*,*) " DEBUG loading from ", fname 
	 open (unit = 69, file = fname, status = 'old')
	 do iline = 1,1000 ! search for BEGIN_DATAGRID_3D_
	 	read (69,*) line
	 	if ( index ( line, "DATAGRID_3D_") .ne. 0 ) exit
	 end do
	 if ( ifile .eq. 1 ) then
		 read(69,*) this%N(:)
		 read(69,*) lvs(:,1)
		 read(69,*) lvs(:,2)
		 read(69,*) lvs(:,3)
		 read(69,*) lvs(:,4)
		 	this%Rmin  (:) = lvs(:,1)
		 	this%Rspan (1) = lvs(1,2)
		 	this%Rspan (2) = lvs(2,3)
		 	this%Rspan (3) = lvs(3,4)
		 	this%Rmax  (:) = this%Rmin(:)  + this%Rspan(:)
		 	this%step  (:) = this%Rspan(:) / this%N(:)
			this%invStep(:) = 1/this%step(:)
  			call this%echoSetup()
		 allocate ( this%f( 3, this%N(1),this%N(2),this%N(3) ) )
	 else ! ( ifile .eq. 1 )
		read(69,*) 
		read(69,*) 
		read(69,*) 
		read(69,*) 
		read(69,*) 
	 end if ! ( ifile .eq. 1 )
	 do k=1, this%N(3)  ! layer 
		do j=1,  this%N(2)
		   do i=1,  this%N(1)
		     read(69,*) this%f(ifile,i,j,k)
		   end do !i
		end do !j
	 end do !k
	 close (69)
 end do ! ifile	
end subroutine grid3Dvec3_fromXSF

! ===== grid3D :: writeXSF
subroutine grid3Dvec3_writeXSF(this, prename )
 implicit none
 ! === variables
    class (grid3Dvec3) :: this
 	character (*) prename 
 	character (40) fname
 	character (1) fnum
 	integer ifile
    integer :: i, j, k
 ! === body
 do ifile = 1,3
	write (fnum,'(i1.1)') ifile
	fname = prename//"_"//fnum//'.xsf'
	write (*,*) " DEBUG writing to ", fname
	open (unit = 69, file = fname, status = 'unknown')
	write(69,*) "CRYSTAL"
	write(69,*) "PRIMVEC"
	write(69,'(3f12.6)')  this%Rspan(1), 0.0, 0.0
	write(69,'(3f12.6)')  0.0, this%Rspan(2), 0.0
	write(69,'(3f12.6)')  0.0, 0.0, this%Rspan(3)
	write(69,*) "CONVVEC"
	write(69,'(3f12.6)')  this%Rspan(1), 0.0, 0.0
	write(69,'(3f12.6)')  0.0, this%Rspan(2), 0.0
	write(69,'(3f12.6)')  0.0, 0.0, this%Rspan(3)
	write(69,*) "PRIMCOORD"
	write(69,*) "1  1"
	write(69,*) "1    0.000000    0.000000    0.00000"
	write(69,*)
	write(69,*) "BEGIN_BLOCK_DATAGRID_3D"
	write(69,*) "density_3D"                    
	write(69,*) " BEGIN_DATAGRID_3D_DENSITY"
	write(69,'(3i10)') this%N(:)
	write(69,'(3f12.6)')  0.0, 0.0, 0.0
	write(69,'(3f12.6)')  this%Rspan(1), 0.0, 0.0
	write(69,'(3f12.6)')  0.0, this%Rspan(2), 0.0
	write(69,'(3f12.6)')  0.0, 0.0, this%Rspan(3)
	do k=1, this%N(3)  ! layer 
		do j=1,  this%N(2)
			do i=1,  this%N(1)
				write(69,*) this%f(ifile,i,j,k)
			end do !i
		end do !j
	end do !k
	write(69,*) "END_DATAGRID_3D"
	write(69,*) "END_BLOCK_DATAGRID_3D"   
	close (69)
 end do ! ifile	
end subroutine grid3Dvec3_writeXSF

! ===== grid3Dvec :: interpolate
subroutine grid3Dvec3_interpolate(this, R, ff )
 implicit none
  ! === variables
  class (grid3Dvec3) :: this
  real, dimension (3), intent (in)     :: R
  real, dimension (3), intent (inout)  :: ff
  integer ix0,iy0,iz0, ix1,iy1,iz1
  real    tx,ty,tz, mx,my,mz
  ! === body
  !write (*,'(6f20.10,3i5)') R(:),this%step(:),this%N(:)
  !tx  = R(1)/this%step(1)
  !ty  = R(2)/this%step(2)
  !tz  = R(3)/this%step(3)
  tx  = R(1)*this%invStep(1)
  ty  = R(2)*this%invStep(2)
  tz  = R(3)*this%invStep(3)
  ix0 = int( tx + fastFloorOffset ) - fastFloorOffset
  iy0 = int( ty + fastFloorOffset ) - fastFloorOffset
  iz0 = int( tz + fastFloorOffset ) - fastFloorOffset
  tx = tx - ix0
  ty = ty - iy0
  tz = tz - iz0
  mx = 1  - tx
  my = 1  - ty
  mz = 1  - tz
  ix1 = modulo( ix0+1 , this%N(1) )+1
  iy1 = modulo( iy0+1 , this%N(2) )+1
  iz1 = modulo( iz0+1 , this%N(3) )+1
  ix0 = modulo( ix0   , this%N(1) )+1
  iy0 = modulo( iy0   , this%N(2) )+1
  iz0 = modulo( iz0   , this%N(3) )+1
  ! write (*,'(6f16.5,9i5)')   R(:), this%step(:), x0,y0,z0,   x1,y1,z1,  this%N(:) 
  ff(:) = ff(:) + &
              mz * ( my * ( mx * this%f(:,ix0,iy0,iz0 )     &
                          + tx * this%f(:,ix1,iy0,iz0 ))    &
                   + ty * ( mx * this%f(:,ix0,iy1,iz0 )     &
                          + tx * this%f(:,ix1,iy1,iz0 )))   &
            + tz * ( my * ( mx * this%f(:,ix0,iy0,iz1 )     &
                          + tx * this%f(:,ix1,iy0,iz1 ))    &
                   + ty * ( mx * this%f(:,ix0,iy1,iz1 )     &
                          + tx * this%f(:,ix1,iy1,iz1 )))
end subroutine grid3Dvec3_interpolate

end module T_grid3Dvec3
