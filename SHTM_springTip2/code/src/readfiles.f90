

   subroutine readspecies ()
        use G_globals
		use T_atoms
     implicit none
  
! Local Variable Declaration and Description
! ===========================================================================
        integer i,j
		real Rij,Eij

! Procedure
! ===========================================================================
        write (*,*) " >> Reading species.dat"

        open (unit = 69, file = "species.dat", status = 'old')
        read (69, *) ntypes
		! write (*,*) ntypes
		allocate ( atypes(ntypes) )
		do i = 1, ntypes
         read (69,*) atypes(i)%Z, atypes(i)%R0, atypes(i)%E0, atypes(i)%symbol 
		 write (*,'(A,i10,3f16.8)') " Z,R0,E0,C6 ", atypes(i)%Z, atypes(i)%R0, atypes(i)%E0
        end do
        close (unit = 69)

        allocate (C6ij (ntypes, ntypes))
        allocate (C12ij(ntypes, ntypes))
		do i = 1, ntypes
			do j = 1, ntypes
				Rij = atypes(i)%R0 + atypes(j)%R0
				Eij = sqrt( atypes(i)%E0 * atypes(j)%E0 )
              C6ij(i,j)  = 2*Eij*(Rij**6) 
			  C12ij(i,j) =   Eij*(Rij**12)
		 	if (wrtDebug .gt. 0) then
				 write (*,'(A,2i5,4f25.10)') " i,j Rij,Eij,C6,C12 ", i,j, Rij, Eij, C6ij(i,j),C12ij(i,j)
			end if
			end do ! j
        end do ! i
		
 end subroutine readspecies


 subroutine system2XYZ ( system, filenum)
   use G_globals
   use T_atoms
 implicit none
  type (subSystem)   , intent(in)  :: system
  integer filenum
  integer i
  !write (*,*) " >> subSystem.toXYZ:  ", filenum
  do i = 1, system%n
  	 write (filenum, '( A, 3f20.8 )') atypes(system%Zs(i))%symbol, system%Rs(:,i)
  end do
 end subroutine system2XYZ


! subroutine system2XYZ ( filenum, n,Zs,Rs )
!   use G_globals
!   use T_atoms
! implicit none
!  ! class (subSystem) system
!  integer filenum
!  integer n	                                     ! number of atoms
!  integer, dimension (:), allocatable :: Zs      ! proton numbers of atoms
!  real, dimension (:, :), allocatable :: Rs      ! positions of atoms
!  integer i
!  write (*,*) " >> subSystem.toXYZ:  ", filenum
!  do i = 1, n
!  	 write (filenum, '( A, 3f20.8 )') atypes(Zs(i))%symbol, Rs(:,i)
!  end do
! end subroutine system2XYZ

