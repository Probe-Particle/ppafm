CC=g++
LD=g++
CPPFLAGS= -fPIC -std=c++11 -O2 -march=native -mtune=native
LDFLAGS= -shared

all: MP RR PolyCycles_ atomfit_ MMFF_ FARFF_ COvsCO_ SimplePot_

MP: Multipoles.o

RR: ReactiveFF.o

PolyCycles_: PolyCycles.o

MMFF_: MMFF.o

FARFF_: FARFF.o

SimplePot_: SimplePot.o

COvsCO_: COvsCO.o

atomfit_: atomfit.o

GLV: GLView.o Draw3D.o
	$(LD) $(LDFLAGS) GLView.o Draw3D.o -o GLV_lib.so

# .o files depend on .cpp files with same prefix
# $@ target, $< prerequisite, $* matched stencil
%.o:: %.cpp
	$(CC) $(CPPFLAGS) -c $< -o $@
	$(LD) $(LDFLAGS) $@ -o $*_lib.so

clean:
	rm -f *.o *.so *.pyc
