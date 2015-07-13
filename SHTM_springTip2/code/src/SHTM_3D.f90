
program SHTM_3D
    use G_globals
  implicit none

! == Local Parameters and Data Declaration
! == Local Variable Declaration and Description
integer ix,iy,iz,OutNtot

real Eprobe ! Etip
real, dimension (3) :: Rtip0Atom1, Rprobe, Fprobe, Rtmp ! Ftip

integer ipos,npos
real, dimension (:,:), allocatable :: poslist

real, dimension (3) :: a1,a2
real R1, TSss,TTss, TSpp,TTpp

! real time_start,time_end
real time_start2,time_end2

real, dimension (3) :: eignums

LOGICAL :: file_exists

integer out_E, out_Fxy, out_Tss, out_Tpp, out_pos, out_vib

type (grid3D)     :: OutE, OutFx, OutFy, OutFz, OutX, OutY, OutZ, OutTTpp,OutTSpp, OutTTss, OutTSss, OutEig1, OutEig2, OutEig3

! == Procedure

a1 = (/ 0,0,1 /)  ! orientation of pz orbitals
! beta = 1.0

! ==================================================
! =================== loading inputs
! ==================================================

write (*,*) " >> Reading SHTM_3D.ini"
  open (unit = 69, file = "SHTM_3D.ini", status = 'old')
  read (69,*) wrtDebug
	write (*,*) "DEBUG 1 "
  read (69,*) relaxMethod, onGrid
	write (*,*) "DEBUG 2 "
  read (69,*) sampleOutside, FgridMax
	write (*,*) "DEBUG 3 "
  FgridMaxSq = FgridMax*FgridMax
  read (69,*) dt, convergF, damping, startKick, maxRelaxIter  
	write (*,*) "DEBUG 4 "
  read (69,*) l0_radial, k_radial
	write (*,*) "DEBUG 5 "
  read (69,*) beta1, beta2
	write (*,*) "DEBUG 6 "
  read (69,*) kHarmonic
	write (*,*) "DEBUG 7 "
  read (69,*) RspringMin
	write (*,*) "DEBUG 8 "
  read (69,*) ddisp
	write (*,*) "DEBUG 9 "
  read (69,*) probeZ, Qprobe
	write (*,*) "DEBUG 10 "
  read (69,*) OutFz%N (:)
	write (*,*) "DEBUG 11 "
  read (69,*) OutFz%R0(:)
	write (*,*) "DEBUG 12 "
  read (69,*) OutFz%cell(:,1)
	write (*,*) "DEBUG 13 "
  read (69,*) OutFz%cell(:,2)
	write (*,*) "DEBUG 14 "
  read (69,*) OutFz%cell(:,3)
	write (*,*) "DEBUG 15 "
  read (69,*) out_E, out_pos, out_Fxy, out_vib, out_Tss, out_Tpp
	write (*,*) "DEBUG 16 "
  close (69)

  OutNtot = OutFz%N(1)*OutFz%N(2)*OutFz%N(3) 

! ==================================================
! =================== initialization of LJ potiential
! ==================================================

  ! atomic parameters
write (*,*) " >> Read atomic positions and parameters "
call readspecies()
call surf%fromfile( 'surf.bas' )
call tip%fromfile ( 'tip.bas'  )

open (unit = 69, file = "poslist.ini", status = 'old')
	read (69,*) npos
	allocate( poslist(3,npos) )
	do ipos = 1,npos
		read (69,*) Rtmp(:)
		poslist(1,ipos) =  (  atypes( tip%Zs(1) )%R0 + atypes( probeZ )%R0  ) * Rtmp(1) * sin( 2*3.14159265359*Rtmp(2) ) * cos( 2*3.14159265359*Rtmp(3) )
		poslist(2,ipos) =  (  atypes( tip%Zs(1) )%R0 + atypes( probeZ )%R0  ) * Rtmp(1) * sin( 2*3.14159265359*Rtmp(2) ) * sin( 2*3.14159265359*Rtmp(3) )
		poslist(3,ipos) = -(  atypes( tip%Zs(1) )%R0 + atypes( probeZ )%R0  ) * Rtmp(1) * cos( 2*3.14159265359*Rtmp(2) )  
		if ( wrtDebug .gt. 0 ) write(*,'(A,i5,6f20.10)') " i, R,theta,phi, pos0 x,y,z: ", ipos,Rtmp(:), poslist(:,ipos)
	end do ! ipos

! ==================================================
! =================== creating GRID FORCE FIELD
! ==================================================


if ( onGrid .eq. 1 ) then
	withElectrostatic = 0
	INQUIRE(FILE="FFelec_3.xsf", EXIST=file_exists)
	if (file_exists) then
		withElectrostatic = 1
		INQUIRE(FILE="FFvdw_3.xsf", EXIST=file_exists)
		if (file_exists) then
			write (*,*) ">> Precomputed classical surface ForceField found"
			write (*,*) ">> Loading Electrostatic Force ..."
			call FFelec%fromXSF( "FFelec" )
			call FFgrid%copySetup(FFelec)	
			FFgrid%f(:,:,:,:)  = FFelec%f(:,:,:,:) * Qprobe
			deallocate( FFelec%f ) 
			write (*,*) ">> Loading van der Waals Force ..."
			call FFvdW%fromXSF( "FFvdw" )
			FFgrid%f(:,:,:,:)  = FFgrid%f(:,:,:,:) + FFvdw%f(:,:,:,:)
			deallocate( FFvdw%f ) 
			write (*,*) ">> Loading Pauli Force ..."
			call FFpauli%fromXSF( "FFpauli" )
			FFgrid%f(:,:,:,:)  = FFgrid%f(:,:,:,:) + FFpauli%f(:,:,:,:)
			deallocate( FFpauli%f ) 
		else
			write (*,*) ">> No precomputed force field => do sampling ..."
			write (*,*) ">> Loading Electrostatic Force ..."
			call FFelec%fromXSF( "FFelec" )
			write (*,*) " DEBUG FFelec setup: ..."
			call FFelec%echoSetup ( )
			call FFgrid%copySetup(FFelec)
			write (*,*) " DEBUG FFgrid setup: ..."
			call FFgrid%echoSetup ( )
			call FFpauli%copySetup(FFelec)
			write (*,*) " DEBUG FFpauli setup: ..."
			call FFpauli%echoSetup ( )
			write (*,*) ">> Sampling surface force field ..."
			write (*,*) " FFgrid%N ", FFgrid%N
			call cpu_time(time_start2)
			call sampleSurf( surf, FFgrid, FFpauli )
			call cpu_time(time_end2)
			write (*,'(A,f20.10)') " time(sampleSurf) total     [sec] : ", (time_end2-time_start2) 
			write (*,'(A,f20.10)') " time(sampleSurf) per_pixel [sec] : ", (time_end2-time_start2)/( FFgrid%N(1)*FFgrid%N(2)*FFgrid%N(3)  )
			call FFgrid %writeXSF( "FFvdw" )
			call FFpauli%writeXSF( "FFpauli" )
			write (*,*) ">> Scaling and adding electrostatic force ..."
			FFgrid%f(:,:,:,:)  = FFgrid%f(:,:,:,:)  +  FFpauli%f(:,:,:,:)  +  FFelec%f(:,:,:,:) * Qprobe
			deallocate( FFelec%f ) ! to save some memory
			deallocate( FFpauli%f )  ! to save some memory
			!write (*,*) ">> writing Total Force Field ..."
			!call FFgrid%writeXSF( "FFgrid" )
		end if ! file_exists
	else	! withElectrostatic
		INQUIRE(FILE="FFvdw_3.xsf", EXIST=file_exists)
		if (file_exists) then
			write (*,*) ">> Precomputed classical surface ForceField found"
			write (*,*) ">> Loading van der Waals Force ..."
			call FFgrid%fromXSF( "FFvdw" )
			write (*,*) ">> Loading Pauli Force ..."
			call FFpauli%fromXSF( "FFpauli" )
			FFgrid%f(:,:,:,:)  = FFgrid%f(:,:,:,:) + FFpauli%f(:,:,:,:)
			deallocate( FFpauli%f ) 
		else
			write (*,*) ">> No precomputed force field => do sampling ..."
			write (*,*) ">> Loading Grid params ..."
			open (unit = 69, file = "grid.ini", status = 'old')
			read (69,*) FFgrid%Rmin(:)
			read (69,*) FFgrid%Rmax(:)
			read (69,*) FFgrid%step(:)
			close ( 69 )
			call FFgrid%initgrid()
			call FFpauli%copySetup(FFgrid)
			call FFpauli%echoSetup ( )
			write (*,*) ">> Sampling surface force field ..."
			call cpu_time(time_start2)
			call sampleSurf( surf, FFgrid, FFpauli )
			call cpu_time(time_end2)
			write (*,'(A,f20.10)') " time(sampleSurf) total     [sec] : ", (time_end2-time_start2) 
			write (*,'(A,f20.10)') " time(sampleSurf) per_pixel [sec] : ", (time_end2-time_start2)/( FFgrid%N(1)*FFgrid%N(2)*FFgrid%N(3)  )
			write (*,*) ">> Saving and adding forcefileds ..."
			call FFgrid %writeXSF( "FFvdw" )
			call FFpauli%writeXSF( "FFpauli" )
			FFgrid%f(:,:,:,:)  = FFgrid%f(:,:,:,:)  +  FFpauli%f(:,:,:,:)
			deallocate( FFpauli%f )  ! to save some memory
		end if ! file_exists
	end if! ! withElectrostatic
else	! onGrid
	write (*,*) ">> Loading Electrostatic Force ..."
	call FFelec%fromXSF( "Felec" )
	write (*,*) ">> Set froce field strength..."
	FFelec%f(:,:,:,:)  = FFelec%f(:,:,:,:) * Qprobe
end if



! ==================================================
! =================== creating output grids
! ==================================================

write (*,*) " >> Initialize Grids over Tip position "
  call OutFz %initgrid()
  call OutFz %echoSetup
  if (out_E .gt.0 ) then
	call OutE    %copySetup(OutFz)
  end if
  if (out_Fxy .gt.0 ) then
    call OutFx   %copySetup(OutFz)
    call OutFy   %copySetup(OutFz)
  end if
  if (out_pos .gt.0 ) then
    call OutX    %copySetup(OutFz)
    call OutY    %copySetup(OutFz)
    call OutZ    %copySetup(OutFz)
  end if
  if (out_Tpp .gt.0 ) then
    call OutTTpp   %copySetup(OutFz)
    call OutTSpp   %copySetup(OutFz)
  end if
  if (out_Tss .gt.0 ) then
    call OutTTss  %copySetup(OutFz)
    call OutTSss  %copySetup(OutFz)
  end if
  if (out_vib .gt.0 ) then
    call OutEig1  %copySetup(OutFz)
    call OutEig2  %copySetup(OutFz)
    call OutEig3  %copySetup(OutFz)
  end if

! ==================================================
! =================== main loop
! ==================================================

Rprobe0(:) = 0
! Rprobe0(3) = - ( atypes( tip%Zs(1) )%R0 + atypes( probeZ )%R0 )
Rprobe0(3) = -l0_radial
write (*,'(A,3f20.5)')  " Rprobe0 normalized :   ",  Rprobe0 

write (*,*) " >> TIP sampling .... "
    !  OutE %f = 0

! performance measurements
call cpu_time(time_start2)
relaxItersSum = 0

Rtip0Atom1(:) = tip%Rs(:,1)

write (*,*) "relaxMethod : ",relaxMethod
write (*,*) "onGrid      : ",onGrid 

do ix = 1, OutFz%N(1)
	do iy = 1, OutFz%N(2)
    	do iz = 1, OutFz%N(3)
			! move tip
			Rtip(:)   = OutFz%R0(:) + (ix-1)*OutFz%step(:,1) + (iy-1)*OutFz%step(:,2) + (iz-1)*OutFz%step(:,3)
			Rtmp(:)   = Rtip(:) - ( tip%Rs(:,1) - Rtip0Atom1(:) )                              
			call tip%move( Rtmp )
			if (iz .eq. 1) then 
				Rprobe(:) = Rtip(:) + Rprobe0(:)
				write (*,'(A,2i10.10,2f20.10)') "ix,iy,Rtip_x,y: ",ix,iy,Rtip(1),Rtip(2)
			else
				Rprobe(:) = Rprobe(:) + OutFz%step(:,3)
			end if
			!write (*,'(3i,9f20.10)') ix,iy,iz, Rtip(:), tip%Rs(:,1), Rprobe(:) 
			! sample
			iiter = 0
			Fprobe(:) = 0
			Eprobe    = 0
			if (relaxMethod .gt. 0) then
				if ( onGrid .eq. 1 ) then
					call relaxGrid( Rprobe, Fprobe, Eprobe  )
				else ! onGrid
					call getFF_LJ       ( Rprobe, probeZ, surf, Eprobe, Fprobe) 
					if (Eprobe .gt. 0) then
						do ipos = 1,npos
							Rprobe(:) = tip%Rs(:,1) + poslist(:,ipos)
							Fprobe(:) = 0
							Eprobe    = 0
							call getFF_LJ ( Rprobe, probeZ, surf, Eprobe, Fprobe)  
							call getFF_LJ ( Rprobe, probeZ,  tip, Eprobe, Fprobe) 
							if (Eprobe .lt. 0) exit
						end do! ipos
					end if ! Eprobe
					call relax    ( Rprobe, Fprobe, Eprobe )  
				end if ! onGrid
			else ! relaxMethod
				Rprobe(:) = Rtip(:) + Rprobe0(:)
				if ( onGrid .eq. 1 ) then
					call FFgrid%interpolate ( Rprobe,Fprobe )
				else
					call FFelec%interpolate ( Rprobe,Fprobe )
					call getFF_LJ ( Rprobe, probeZ, surf, Eprobe, Fprobe)
				end if
			end if ! relaxMethod
			if (wrtDebug .gt. 0) write (*,'(A,4i5,4f25.10)') " ix,iy,iz,iters,E,Fz,RzTip,Rzprobe ",ix,iy,iz, iiter, Rtip(3), Rprobe(3), Eprobe, Fprobe(3)		
			OutFz%f(ix,iy,iz)   = Fprobe(3)
			if (out_E .gt.0 ) then
				OutE%f(ix,iy,iz)    = Eprobe
			end if ! out_E 
			if (out_Fxy .gt.0 ) then
				OutFx%f(ix,iy,iz)   = Fprobe(1)
				OutFy%f(ix,iy,iz)   = Fprobe(2)
			end if ! out_Fxy
			if (out_pos .gt.0 ) then
				OutX%f(ix,iy,iz)    = Rprobe(1)-Rtip(1)-Rprobe0(1)
				OutY%f(ix,iy,iz)    = Rprobe(2)-Rtip(2)-Rprobe0(2)
				OutZ%f(ix,iy,iz)    = Rprobe(3)-Rtip(3)-Rprobe0(3)
			end if ! out_pos
			if (out_Tpp .gt.0 ) then
				a2(:) = Rprobe(:) - tip%Rs(:,1)
				R1    = sqrt(dot_product(a2,a2))
				a2(:) = a2(:)/R1
				call getHoppingPP ( beta1, Rprobe, a1, a2,  tip,  TTpp )
				call getHoppingPP ( beta2, Rprobe, a1, a2,  surf, TSpp )
				OutTTpp%f(ix,iy,iz)   = TTpp
				OutTSpp%f(ix,iy,iz)   = TSpp
			end if ! out_Tpp
			if (out_Tss .gt.0 ) then
				call getHoppingSS ( beta1, Rprobe,          tip,  TTss )
				call getHoppingSS ( beta2, Rprobe,          surf, TSss )
				OutTTss%f(ix,iy,iz)  = TTss
				OutTSss%f(ix,iy,iz)  = TSss
			end if ! out_Tss
			if (out_vib .gt.0 ) then
				if ( onGrid .gt. 0 ) then
					call  dynmatGrid ( surf, Rprobe, eignums )
				else
					call  dynmat     ( surf, Rprobe, eignums )
				end if
				OutEig1%f(ix,iy,iz)  = eignums(1)
				OutEig2%f(ix,iy,iz)  = eignums(2)
				OutEig3%f(ix,iy,iz)  = eignums(3)
			end if ! out_vib
		end do ! iz
    end do ! iy
end do ! ix

call cpu_time(time_end2)
 close(70) 
 close(71) 
write (*,'(A,i10)'    ) " number of tip positions (grid points) : ", ( OutNtot )
write (*,'(A,i10)'    ) " Relaxation Iterations ( total )       : ", ( relaxItersSum )
write (*,'(A,f16.8)'  ) " Relaxation Iterations ( per tip pos ) : ", ( relaxItersSum/real(OutNtot) )
write (*,'(A,f16.8,A)') " CPU_time ( total )        [sec]       : ", (time_end2-time_start2) 
write (*,'(A,f16.8,A)') " CPU_time ( per tip pos )  [sec]       : ", (time_end2-time_start2)/OutNtot

!call OutE %output("OutE" ,format_Out,1.0)

  call OutFz   %writeXSF ("OutFz.xsf")
  if (out_E .gt.0 ) then
	call OutE    %writeXSF ("OutE.xsf")
  end if
  if (out_Fxy .gt.0 ) then
	call OutFx   %writeXSF ("OutFx.xsf")
	call OutFy   %writeXSF ("OutFy.xsf")
  end if
  if (out_pos .gt.0 ) then
	call OutX    %writeXSF ("OutX.xsf")
	call OutY    %writeXSF ("OutY.xsf")
	call OutZ    %writeXSF ("OutZ.xsf")
  end if
  if (out_Tpp .gt.0 ) then
	call OutTTpp   %writeXSF ("OutTTpp.xsf")
	call OutTSpp   %writeXSF ("OutTTpp.xsf")
  end if
  if (out_Tss .gt.0 ) then
	call OutTTss  %writeXSF ("OutTTss.xsf")
	call OutTSss  %writeXSF ("OutTTpp.xsf")
  end if
  if (out_vib .gt.0 ) then
	call OutEig1  %writeXSF ("OutEig1.xsf")
	call OutEig2  %writeXSF ("OutEig2.xsf")
	call OutEig3  %writeXSF ("OutEig3.xsf")
  end if

stop

end program SHTM_3D
