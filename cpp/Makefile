CC=g++
LD=g++
#CPPFLAGS= -fPIC -std=c++11 -Og
#CPPFLAGS= -fPIC -std=c++11 -O3 -ffast-math -ftree-vectorize 
#CPPFLAGS= -fPIC -std=c++11 -O2 -march=native -mtune=native
#CPPFLAGS= -fPIC -std=c++11 -O3 -march=native -mtune=native
CPPFLAGS= -fPIC -std=c++11 -Ofast -march=native -mtune=native -fopenmp -lgomp -g
LDFLAGS= -shared

all: PP GU MP RR PolyCycles_ atomfit_ fitting_ fitSpline_ MMFF_ FARFF_ landauer landauer_new

PP: ProbeParticle.o

GU: GridUtils.o

MP: Multipoles.o

RR: ReactiveFF.o

Ram: Ramman.o

chrings: ChargeRings.o

landauer: LandauerQD.o

landauer_new: LandauerQD_new.o

PolyCycles_: PolyCycles.o

MMFF_: MMFF.o

FARFF_: FARFF.o

atomfit_: atomfit.o

fitting_: fitting.o

fitSpline_: fitSpline.o

GLV: GLView.o Draw3D.o
	$(LD) $(LDFLAGS) GLView.o Draw3D.o -o GLV_lib.so


# .o files depend on .cpp files with same prefix
# $@ target, $< prerequisite, $* matched stencil
%.o:: %.cpp
	$(CC) $(CPPFLAGS) -c $< -o $@
	$(LD) $(LDFLAGS) $@ -o $*_lib.so

clean:
	rm -f *.o *.so *.pyc
