module G_globals
	use T_atoms
	use T_grid3D
	use T_grid3Dvec3
  implicit none

! =========== Definition of system and force field
type (subSystem)     :: surf, tip
real, dimension (3)  :: Rtip,Rprobe0

real    :: Qprobe
integer :: probeZ                                            ! proton number of probe
integer ntypes                                              ! number of atom types 
type (atomType),allocatable        :: atypes(:)             ! list of atom types
real, dimension (:,:), allocatable :: C6ij, C12ij,E0ij      ! precomputed table of pairs coefs

real 				:: ddisp        ! displacement in dynamical matrix
real                :: kMorse       ! stiffness constant of harmoni
real, dimension (3) :: kHarmonic    ! stiffness constant of harmonic potential
real, dimension (3) :: RspringMin   ! stiffness constant of harmonic potential

real k_radial
real l0_radial

! =========== FIRE globals
real FIRE_finc, FIRE_fdec, FIRE_falpha, FIRE_Nmin, FIRE_dtmax, FIRE_dt, FIRE_acoef0, FIRE_acoef

! =========== Relaxation criterium 
real 	 startKick      ! initialize velocity by    v0 = startKick*F*dt
real     damping        ! damp velocity   v = v * (1.0-damping) 
real     dt
real     convergF
integer  maxRelaxIter, iiter
integer  relaxMethod
integer  onGrid, sampleOutside, withElectrostatic

!real betaPi, betaSigma   ! STM decay 
real beta1, beta2  ! STM decay 

integer format_Out  ! set output format for grid3D variables ( bitwise test : 0=none 1=ppm 2=xsf 3=both ) 
integer wrtDebug

! =========== Force Field grids
type (grid3Dvec3) :: FFelec, FFgrid, FFvdW, FFpauli
real FgridMax  , FgridMaxSq            ! limit force field maximum ( good for plotting potential )


! =========== performance testing
integer relaxItersSum

end module G_globals
