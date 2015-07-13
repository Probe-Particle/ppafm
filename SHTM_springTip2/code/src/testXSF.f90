
program testXSF
    use T_grid3D
  implicit none
! == Local Variable Declaration and Description
	type (grid3D) :: V
! == Procedure
  call V%fromXSF("NaCl_Xe_pot_small.xsf")
  call V%writeXSF("out.xsf")
stop

end program testXSF
