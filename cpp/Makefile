CC=g++
LD=g++
#CPPFLAGS= -fPIC -std=c++11 -Og
#CPPFLAGS= -fPIC -std=c++11 -O3 -ffast-math -ftree-vectorize 
CPPFLAGS= -fPIC -std=c++11 -O2 -march=native -mtune=native
#CPPFLAGS= -fPIC -std=c++11 -O3 -march=native -mtune=native
#CPPFLAGS= -fPIC -std=c++11 -Ofast -march=native -mtune=native
LDFLAGS= -shared

all: PP GU MP RR PolyCycles_ atomfit_ fitting_ fitSpline_ MMFF_ FARFF_ COvsCO_ SimplePot_

PP: ProbeParticle.o

GU: GridUtils.o

MP: Multipoles.o

RR: ReactiveFF.o

PolyCycles_: PolyCycles.o

MMFF_: MMFF.o

FARFF_: FARFF.o

SimplePot_: SimplePot.o

COvsCO_: COvsCO.o

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
