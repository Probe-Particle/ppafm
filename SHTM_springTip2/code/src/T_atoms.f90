module T_atoms
implicit none

type atomType
 integer Z        ! proton number
 real    R0       ! vdW radius
 real    E0       ! Morse
 real    C6       ! vdW C6
 real    Q        ! charge
 real    alfa0    ! polarizability
 character (len=3) symbol
end type atomType

type subSystem
    integer :: n	! number of atoms
	integer, dimension (:), allocatable :: Zs      ! proton numbers of atoms
	real, dimension (:, :), allocatable :: Rs      ! positions of atoms
	contains
      procedure :: fromfile
	  procedure :: move
end type subSystem

contains 

! ===== subSystem :: fromfile ========
 subroutine fromfile ( this, fname)
 implicit none
  class (subSystem) :: this
  character (*) fname
  integer i
  write (*,*) " >> subSystem.fromfile:  ", fname
  open (unit = 69, file = fname, status = 'old')
  read (69, *) this%n
  allocate (this%Zs(this%n))
  allocate (this%Rs(3,this%n))
  do i = 1, this%n
  	 read (69,*) this%Zs(i), this%Rs(:,i)
  end do
  close (69)
 end subroutine fromfile

! ===== subSystem :: move ========
 subroutine move ( this, R)
 implicit none
  class (subSystem) :: this
  real, dimension (3), intent(in) :: R
  integer i
  do i = 1, this%n
  	 this%Rs(:,i) = this%Rs(:,i) + R(:)
  end do
 end subroutine move

end module T_atoms




!  Stefan Grimme, 2011  :  Density functional theory with London dispersion corrections,
!     DOI: 10.1002/wcms.30
!  Alexandre Tkatchenko and Matthias Scheffler:  Accurate Molecular Van Der Waals Interactions from Ground-State Electron Density and Free-Atom Reference Data
!     DOI: 10.1103/PhysRevLett.102.073005
!   
!   C6ij = 3/2  f0i*f0j/(f0i+f0j) * alfa0i * alfa0j
!   f0     ... plasma frequency    f0 ~ sqrt( rho ) for jelium
!   alfa0  ... static polariability
!
