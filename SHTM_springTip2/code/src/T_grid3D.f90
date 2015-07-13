module T_grid3D
	
	implicit none

integer, PARAMETER :: fastFloorOffset = 10000    

type grid3D                                     ! 3D grid maps of observables
  real,    dimension (3)   :: R0                ! grid size and spacing (x,y,z)
  real,    dimension (3,3) :: cell,step 
  ! real,    dimension (3,3) :: icell           !  ?
  integer, dimension (3)   :: N                 ! dimension in x,y,z
  real, dimension (:, :, :), allocatable :: f   ! array storing values of othe observable
  contains
    procedure :: initgrid
    procedure :: copySetup
    procedure :: echoSetup
    procedure :: writeXSF  
    procedure :: fromXSF
	! procedure :: interpolate => grid3D_interpolate
end type grid3D

contains 

! ===== grid3D :: initgrid ========
 subroutine initgrid ( this )
 implicit none
  class (grid3D) :: this
  !inv3D (cell,icell)
  this%step(:,1) = this%cell(:,1)/this%N(1)
  this%step(:,2) = this%cell(:,2)/this%N(2)
  this%step(:,3) = this%cell(:,3)/this%N(3)
  allocate (this%f( this%N(1),this%N(2),this%N(3) ))
  this%f(:,:,:) = 0
 end subroutine initgrid

! ===== grid3D :: setByTemplate ========
subroutine copySetup ( this, from )
 implicit none
  class (grid3D) :: this
  class (grid3D) :: from
  this%R0    = from%R0
  this%cell  = from%cell
  this%N     = from%N
  call this%initgrid()
end subroutine copySetup

! ===== grid3D :: echoSetup ========
subroutine echoSetup ( this )
 implicit none
  class (grid3D) :: this
   write (*,'(A,3i10)')   " N:  ", this%N(:)
   write (*,'(A,3f16.8)') " R0: ", this%R0(:)
   write (*,*) "cell:"
   write (*,'(3f16.8)') this%cell(:,1)
   write (*,'(3f16.8)') this%cell(:,2)
   write (*,'(3f16.8)') this%cell(:,3)
   write (*,*) "step:"
   write (*,'(3f16.8)') this%step(:,1)
   write (*,'(3f16.8)') this%step(:,2)
   write (*,'(3f16.8)') this%step(:,3)
   !write (*,*) "icell:"
   !write (*,'(3f16.8)') this%step (:,1)
   !write (*,'(3f16.8)') this%step (:,2)
   !write (*,'(3f16.8)') this%step (:,3)
end subroutine echoSetup

! ===== grid3D :: writeXSF ========
subroutine writeXSF(this, fname)
 implicit none
    class (grid3D) :: this
    character (*) fname 
    integer :: i, j, k
    write (*,*) "write XSF: ",fname
    open (unit = 69, file = fname, status = 'unknown')
 write(69,*) "CRYSTAL"
 write(69,*) "PRIMVEC"
 write(69,'(3f12.6)')  this%cell(:,1)
 write(69,'(3f12.6)')  this%cell(:,2)
 write(69,'(3f12.6)')  this%cell(:,3)
 write(69,*) "CONVVEC"
 write(69,'(3f12.6)')  this%cell(:,1)
 write(69,'(3f12.6)')  this%cell(:,2)
 write(69,'(3f12.6)')  this%cell(:,3)
 write(69,*) "PRIMCOORD"
 write(69,*) "1  1"
 write(69,*) "1    0.000000    0.000000    0.00000"
 write(69,*)
 write(69,*) "BEGIN_BLOCK_DATAGRID_3D"
 write(69,*) "density_3D"                    
 write(69,*) " BEGIN_DATAGRID_3D_DENSITY"
 write(69,'(3i10)')    this%N(:)
 write(69,'(3f12.6)')  this%R0(:)
 write(69,'(3f12.6)')  this%cell(:,1)
 write(69,'(3f12.6)')  this%cell(:,2)
 write(69,'(3f12.6)')  this%cell(:,3)
 do k=1, this%N(3)  ! layer 
    do j=1,  this%N(2)
       do i=1,  this%N(1)
         write(69,*) this%f(i,j,k)
       end do !i
    end do !j
 end do !k
 write(69,*) "END_DATAGRID_3D"
 write(69,*) "END_BLOCK_DATAGRID_3D"   
 close (69)
end subroutine writeXSF

! ===== grid3D :: fromXSF =====
subroutine fromXSF(this, fname)
 implicit none
 class (grid3D) :: this
 character (*) fname 
 integer :: i, j, k
 integer iline
 character (100) line 
! real,    dimension (3,4) :: lvs 
 open (unit = 69, file = fname, status = 'old')
 do iline = 1,1000 ! search for BEGIN_DATAGRID_3D_
 	read (69,*) line
 	if ( index ( line, "BEGIN_DATAGRID_3D_") .ne. 0 ) exit
 end do
 read(69,*) this%N(:)
 read(69,*) this%R0  (:)
 read(69,*) this%cell(:,1)
 read(69,*) this%cell(:,2)
 read(69,*) this%cell(:,3)
 call this%initgrid()
 do k=1, this%N(3)  
    do j=1,  this%N(2)
       do i=1,  this%N(1)
         read(69,*) this%f(i,j,k)
       end do !i
    end do !j
 end do !k
 close (69)
end subroutine fromXSF


end module T_grid3D
