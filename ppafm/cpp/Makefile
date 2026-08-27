CC=g++
LD=g++
MCC=clang++-22
MLD=clang++-22

#CPPFLAGS= -fPIC -std=c++11 -O2 -mtune=native
CPPFLAGS= -fPIC -std=c++11 -O2 -mtune=native -fopenmp
#CPPFLAGS= -fPIC -std=c++11 -Og -g
#CPPFLAGS= -fPIC -std=c++11 -Og -g -fsanitize=address -fno-omit-frame-pointer

LDFLAGS= -shared -fopenmp


all: PP GU fitting_ fitSpline_

PP: ProbeParticle.o

GU: GridUtils.o

fitting_: fitting.o

fitSpline_: fitSpline.o

# MAC:

Mall: MPP MGU Mfitting_ MfitSpline_

MPP: CC=$(MCC)
MPP: LD=$(MLD)
MPP: ProbeParticle.o

MGU: CC=$(MCC)
MGU: LD=$(MLD)
MGU: GridUtils.o

Mfitting_: CC=$(MCC)
Mfitting_: LD=$(MLD)
Mfitting_: fitting.o

MfitSpline_: CC=$(MCC)
MfitSpline_: LD=$(MLD)
MfitSpline_: fitSpline.o

# .o files depend on .cpp files with same prefix
# $@ target, $< prerequisite, $* matched stencil
%.o:: %.cpp
	$(CC) $(CPPFLAGS) -c $< -o $@
	$(LD) $(LDFLAGS) $@ -o $*_lib.so

clean:
	rm -f *.o *.so *.pyc
