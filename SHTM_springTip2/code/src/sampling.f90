
subroutine sampleSurf( system, FF_vdW, FF_pauli )
	use G_globals
	implicit none
! === Parameters
!    abstract interface
!      subroutine getFFsurf ( R,  Z, from, E, F )
!		  real, dimension (3), intent(in)    :: R
!		  integer            , intent(in)    :: Z
!		  type (subSystem)   , intent(in)    :: from
!		  real,                intent(inout) :: E
!		  real, dimension (3), intent(inout) :: F
!     end subroutine getFFsurf
!    end interface
!    procedure (getFFsurf), pointer :: getFFsurf => null ()
	type (subSystem)   , intent(in)    :: system
	type (grid3Dvec3)  , intent(inout) :: FF_vdW
	type (grid3Dvec3)  , intent(inout) :: FF_pauli
! === variables
	integer ix,iy,iz
	real    Epauli, EvdW    ! Eprobe 
	real, dimension (3) :: Rprobe, Fpauli, Fvdw, Rshift   ! Fprobe
	real, dimension (3) :: tmp
	real fmag2
! === Body
!	write (*,*) " DEBUG scaleFF : ", scaleFF

	write (*,*) " DEBUG FF_vdW%N ", FF_vdW%N
	call FF_vdW%echoSetup ( )

	do ix = 1, FF_vdW%N(1)
		Rprobe(1) = FF_vdW%Rmin(1) + (ix-1) * FF_vdW%step(1)
		write (*,*) " ix: ", ix
		do iy = 1, FF_vdW%N(2)         
			Rprobe(2) = FF_vdW%Rmin(2) + (iy-1) * FF_vdW%step(2)                     
			do iz = 1, FF_vdW%N(3)
				Rprobe(3) = FF_vdW%Rmin(3) + (iz-1) * FF_vdW%step(3)
				! Fprobe(:) = 0.0 
				Fpauli(:) = 0.0
				FvdW(:)   = 0.0
				tmp = Rprobe
				! call getFF_LJ    ( tmp, probeZ, system, Eprobe, Fprobe )
				call getFF_Pauli ( tmp, probeZ, system, Epauli, Fpauli )
				call getFF_vdW   ( tmp, probeZ, system, EvdW  , FvdW   )
				if ( sampleOutside .gt. 0 ) then                         ! sample neighboring cells
					Rshift(:) = 0.D0
					Rshift(1) = FF_vdW%Rspan(1) 
					tmp = Rprobe + Rshift
					! call getFF_LJ  ( tmp, probeZ, system, Eprobe, Fprobe )
					call getFF_Pauli ( tmp, probeZ, system, Epauli, Fpauli )
					call getFF_vdW   ( tmp, probeZ, system, EvdW  , FvdW   )
					tmp = Rprobe - Rshift
					! call getFF_LJ  ( tmp, probeZ, system, Eprobe, Fprobe )
					call getFF_Pauli ( tmp, probeZ, system, Epauli, Fpauli )
					call getFF_vdW   ( tmp, probeZ, system, EvdW  , FvdW   )
					Rshift(:) = 0.D0
					Rshift(2) = FF_vdW%Rspan(2) 
					tmp =  Rprobe + Rshift
					! call getFF_LJ  ( tmp, probeZ, system, Eprobe, Fprobe )
					call getFF_Pauli ( tmp, probeZ, system, Epauli, Fpauli )
					call getFF_vdW   ( tmp, probeZ, system, EvdW  , FvdW   )
					tmp = Rprobe - Rshift
					! call getFF_LJ  ( tmp, probeZ, system, Eprobe, Fprobe )
					call getFF_Pauli ( tmp, probeZ, system, Epauli, Fpauli )
					call getFF_vdW   ( tmp, probeZ, system, EvdW  , FvdW   )
					Rshift(:) = 0.D0
					Rshift(1) = FF_vdW%Rspan(1) 
					Rshift(2) = FF_vdW%Rspan(2) 
					tmp = Rprobe + Rshift
					! call getFF_LJ  ( tmp, probeZ, system, Eprobe, Fprobe )
					call getFF_Pauli ( tmp, probeZ, system, Epauli, Fpauli )
					call getFF_vdW   ( tmp, probeZ, system, EvdW  , FvdW   )
					tmp = Rprobe - Rshift
					! call getFF_LJ  ( tmp, probeZ, system, Eprobe, Fprobe )
					call getFF_Pauli ( tmp, probeZ, system, Epauli, Fpauli )
					call getFF_vdW   ( tmp, probeZ, system, EvdW  , FvdW   )
					Rshift(:) = 0.D0
					Rshift(1) =  FF_vdW%Rspan(1) 
					Rshift(2) = -FF_vdW%Rspan(2) 
					tmp = Rprobe + Rshift
					! call getFF_LJ  ( tmp, probeZ, system, Eprobe, Fprobe )
					call getFF_Pauli ( tmp, probeZ, system, Epauli, Fpauli )
					call getFF_vdW   ( tmp, probeZ, system, EvdW  , FvdW   )
					tmp = Rprobe - Rshift
					! call getFF_LJ  ( tmp, probeZ, system, Eprobe, Fprobe )
					call getFF_Pauli ( tmp, probeZ, system, Epauli, Fpauli )
					call getFF_vdW   ( tmp, probeZ, system, EvdW  , FvdW   )
				end if
				!fmag2 = dot_product( Fprobe, Fprobe )
				!if (fmag2 .gt. FgridMaxSq ) Fprobe(:) = Fprobe(:)*sqrt(FgridMaxSq/fmag2)   ! just to remove very high numbers from data grids
				! FF%f(:,ix,iy,iz) = FF%f(:,ix,iy,iz) + scaleFF * Fprobe(:)

				fmag2 = dot_product( Fpauli, Fpauli )
				if (fmag2 .gt. FgridMaxSq ) Fpauli(:) = Fpauli(:)*sqrt(FgridMaxSq/fmag2)   ! just to remove very high numbers from data grids
				FF_pauli %f(:,ix,iy,iz) = Fpauli(:)

				fmag2 = dot_product( Fvdw, Fvdw )
				if (fmag2 .gt. FgridMaxSq ) Fvdw(:) = Fvdw(:)*sqrt(FgridMaxSq/fmag2)   ! just to remove very high numbers from data grids
				FF_vdw   %f(:,ix,iy,iz) = Fvdw(:)
				! FF%f(:,ix,iy,iz) = scaleFF
			end do ! iz
		end do ! iy
	end do ! ix
end subroutine sampleSurf
