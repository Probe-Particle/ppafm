
program SHTM_1D
    use G_globals
  implicit none

! == Local Parameters and Data Declaration
! == Local Variable Declaration and Description
integer ix,iy,iz,nz

real xscan,yscan, zmin, zmax, zstep

real Esond, Etip
real, dimension (3) :: Rtip,Rsond, Fsond, Rtmp, Rsond0, Ftip

integer ipos,npos
real, dimension (:,:), allocatable :: poslist

real time_start,time_end

! == Procedure

write (*,*) " >> Reading SHTM_1D.ini"
  open (unit = 69, file = "SHTM_1D.ini", status = 'old')
  read (69,*) wrtDebug
  read (69,*) relaxMethod
  read (69,*) dt, convergF, damping,startKick, maxRelaxIter  
  read (69,*) kMorse 
  read (69,*) kHarmonic
  read (69,*) RspringMin
  read (69,*) sondZ
  read (69,*) xscan, yscan
  read (69,*) zmax, zmin, zstep
  close (69)

nz = floor((zmax - zmin)/zstep)

  ! atomic parameters
write (*,*) " >> Read atomic positions and parameters "
call readspecies()
call surf%fromfile( 'surf.bas' )
call tip%fromfile ( 'tip.bas'  )
Rsond0  = 0
Rsond0(3) =  - ( atypes( tip%Zs(1) )%R0 + atypes( SondZ )%R0 )
write (*,'(A,3f20.5)')  " Rsond0 normalized :   ",  Rsond0 


open (unit = 69, file = "poslist.ini", status = 'old')
	read (69,*) npos
	allocate( poslist(3,npos) )
	do ipos = 1,npos
		read (69,*) Rtmp(:)
		poslist(1,ipos) =  (  atypes( tip%Zs(1) )%R0 + atypes( SondZ )%R0  ) * Rtmp(1) * sin( 2*3.14159265359*Rtmp(2) ) * cos( 2*3.14159265359*Rtmp(3) )
		poslist(2,ipos) =  (  atypes( tip%Zs(1) )%R0 + atypes( SondZ )%R0  ) * Rtmp(1) * sin( 2*3.14159265359*Rtmp(2) ) * sin( 2*3.14159265359*Rtmp(3) )
		poslist(3,ipos) = -(  atypes( tip%Zs(1) )%R0 + atypes( SondZ )%R0  ) * Rtmp(1) * cos( 2*3.14159265359*Rtmp(2) )  
		if ( wrtDebug .gt. 0 ) write(*,'(A,i5,6f20.10)') " i, R,theta,phi, pos0 x,y,z: ", ipos,Rtmp(:), poslist(:,ipos)
	end do ! ipos


write (*,*) " >> TIP sampling .... "
open (unit = 101, file = "1D_E.dat", status = 'unknown')
open (unit = 102, file = "1D_F.dat", status = 'unknown')
open (unit = 103, file = "1D_R.dat", status = 'unknown')
open (unit = 104, file = "1D_dR.dat", status = 'unknown')

open (unit = 200, file = "1D_geom.xyz", status = 'unknown')

Rtip(1) = xscan
Rtip(2) = yscan
Rtip(3) = zmax
!Rsond   = Rtip

Rtmp(:) = Rtip(:) - tip%Rs(:,1)                                  ! move tip to pos first atom to Rtip
call tip%move( Rtmp )
Rsond(:) = tip%Rs(:,1) + Rsond0

call cpu_time(time_start)
write (*,'(A,3f20.10)') " tip%Rs ",tip%Rs(:,1)
write (*,'(A,3f20.10)') " Rsond  ",Rsond(:)
do iz = 1, nz
	iiter = 0
	Rtmp(:) = Rtip(:) - tip%Rs(:,1)                                  ! move tip to pos first atom to Rtip
	call tip%move( Rtmp )
	Etip = 0.0
	Ftip = 0.0
	call getFF_LJ  ( tip%Rs(:,1), tip%Zs(1), surf, Etip, Ftip )
	!Rsond(:) = tip%Rs(:,1) + Rsond0
	iiter = 0
	if (relaxMethod .gt. 0) then
		Fsond(:) = 0
		Esond    = 0
		call getFF_LJ       ( Rsond, sondZ, surf, Esond, Fsond)  ! Surface potential
		!call getFF_LJ       ( Rsond, sondZ,    tip, Esond, Fsond)  ! Tip poteitnal
		if (Esond .gt. 0) then
			do ipos = 1,npos
				Rsond(:) = tip%Rs(:,1) + poslist(:,ipos)
				Fsond(:) = 0
				Esond    = 0
				call getFF_LJ       ( Rsond, sondZ, surf, Esond, Fsond)  ! Surface potential
				call getFF_LJ       ( Rsond, sondZ,  tip, Esond, Fsond)  ! Tip poteitnal
				if (Esond .lt. 0) exit
			end do! ipos
		end if ! Esond
		call relax( surf, Rtip,  Rsond, Fsond, Esond )  ! this is the importaint
		! stop
	else
		Rsond(:) = Rtip(:) + Rsond0
		Fsond(:) = 0
		Esond    = 0
		call getFF_LJ  ( Rsond, sondZ, surf, Esond, Fsond )
	end if
	if ( dot_product(Rsond,Rsond) .gt. 10000.0 ) then
  		Rsond = 0
		Fsond = 0
		Esond = 0
	end if
	if (wrtDebug .gt. 0) then
		write (200,'(i5)')   (1+tip%n + surf%n) 
		write (200,'(A,2i5)') "  iz, iteration ", iz,iiter
		write (200, '( A, 3f20.8 )') atypes(sondZ)%symbol, Rsond
		call system2XYZ ( tip, 200)
		call system2XYZ ( surf, 200)
	end if ! wrtDebug 
	write (*,'(A,2i5,6f25.10)') " iz,iters,E,Fz,RzTip,RzSond ",iz, iiter, Rtip(3),tip%Rs(3,1),Rtmp(3), Rsond(3), Esond, Fsond(3)
	write (101,'(3f20.10)')    Rtip(3), Etip, Esond
	write (102,'(7f20.10)')    Rtip(3), Ftip, Fsond
	write (103,'(7f20.10)')    Rtip(3), Rsond
	write (104,'(4f20.10)')    Rtip(3), (Rsond-Rtip)
	Rtip(3) = Rtip(3) - zstep
end do ! iz
call cpu_time(time_end)
write (*,'(A,f10.5,A)') " >>  .... done in  ", (time_end-time_start) ,"[sec]"
write (*,'(A,f10.5,A)') " >>  = ", (time_end-time_start)/nz,"[sec] per tip step "

 close(101) 
 close(102) 
 close(103) 
 close(104) 

 write (200,*)
 close(200) 

stop

end program SHTM_1D
