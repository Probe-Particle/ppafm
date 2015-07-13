#FC      = ifort
#FFLAGS  = -r8 -g -debug -check bounds -check all -warn all -gen-interfaces -warn interfaces                #  debug
#FFLAGS  = -r8 -g -debug -check bounds -check all               #  debug
#FFLAGS = -r8 -O3 -fno-alias -unroll -static                   #  optimized
#FFLAGS = -r8 -O3 -fno-alias -unroll -static -pg -vec-report2  #  profiler


FC = gfortran
#FFLAGS = -ffree-form -ffree-line-length-0 -fdefault-real-8 -g -O0                    #  debug     gfortran
#FFLAGS = -ffree-form -ffree-line-length-0 -fdefault-real-8 -O3                       #  optimized gfortran
FFLAGS = -ffree-form -ffree-line-length-0 -fdefault-real-8 -Ofast -fstrict-aliasing   #  optimized gfortran



OBJECTS    = readfiles.o forceFields.o relax.o dynmat.o sampling.o
GLOBALS    = G_globals.o eigenSolver.o 
TYPES      = T_grid3D.o T_grid3Dvec.o T_atoms.o
MODULES    = $(TYPES) $(GLOBALS) 

# ==== TARGETS =====

all : SHTM_3D

clean:
	rm -f -r core *.o *.mod *.x

SHTM_3D: SHTM_3D.o $(OBJECTS)
	$(FC) $(FFLAGS) -o SHTM_3D.x SHTM_3D.o $(OBJECTS) $(MODULES)

SHTM_3D_pos0: SHTM_3D_pos0.o $(OBJECTS)
	$(FC) $(FFLAGS) -o SHTM_3D_pos0.x SHTM_3D_pos0.o $(OBJECTS) $(MODULES)

SHTM_1D: SHTM_1D.o $(OBJECTS)
	$(FC) $(FFLAGS) -o SHTM_1D.x SHTM_1D.o $(OBJECTS) $(MODULES)

SHTM_1D_pos0: SHTM_1D_pos0.o $(OBJECTS)
	$(FC) $(FFLAGS) -o SHTM_1D_pos0.x SHTM_1D_pos0.o $(OBJECTS) $(MODULES)

testEigen: testEigen.o $(GLOBALS)
	$(FC) $(FFLAGS) -o testEigen.x testEigen.o $(GLOBALS)

testXSF: testXSF.o $(MODULES)
	$(FC) $(FFLAGS) -o testXSF.x testXSF.o $(MODULES)

# ====  TYPES =====

T_atoms.o : src/T_atoms.f90
	$(FC) $(FFLAGS) -c src/T_atoms.f90
T_grid3D.o  : src/T_grid3D.f90
	$(FC) $(FFLAGS) -c src/T_grid3D.f90
T_grid3Dvec.o  : src/T_grid3Dvec.f90 T_grid3D.o
	$(FC) $(FFLAGS) -c src/T_grid3Dvec.f90


# ==== GLOBALS =====

eigenSolver.o : src/eigenSolver.f90 $(TYPES) 
	$(FC) $(FFLAGS) -c src/eigenSolver.f90

G_globals.o : src/G_globals.f90 $(TYPES) 
	$(FC) $(FFLAGS) -c src/G_globals.f90

# ==== OBJECTS =====

sampling.o : src/sampling.f90 $(MODULES) 
	$(FC) $(FFLAGS) -c src/sampling.f90
relax.o : src/relax.f90 $(MODULES) 
	$(FC) $(FFLAGS) -c src/relax.f90
forceFields.o : src/forceFields.f90 $(MODULES)
	$(FC) $(FFLAGS) -c src/forceFields.f90
readfiles.o : src/readfiles.f90 $(MODULES)
	$(FC) $(FFLAGS) -c src/readfiles.f90
dynmat.o : src/dynmat.f90 $(MODULES)
	$(FC) $(FFLAGS) -c src/dynmat.f90

# ==== PROGRAMS =====

SHTM_3D.o : src/SHTM_3D.f90 $(MODULES) 
	$(FC) $(FFLAGS) -c src/SHTM_3D.f90
SHTM_3D_pos0.o : src/SHTM_3D_pos0.f90 $(MODULES) 
	$(FC) $(FFLAGS) -c src/SHTM_3D_pos0.f90
SHTM_1D.o : src/SHTM_1D.f90 $(MODULES) 
	$(FC) $(FFLAGS) -c src/SHTM_1D.f90
SHTM_1D_pos0.o : src/SHTM_1D_pos0.f90 $(MODULES) 
	$(FC) $(FFLAGS) -c src/SHTM_1D_pos0.f90

testEigen.o : src/testEigen.f90 $(MODULES) 
	$(FC) $(FFLAGS) -c src/testEigen.f90
testXSF.o : src/testXSF.f90 $(MODULES) 
	$(FC) $(FFLAGS) -c src/testXSF.f90


