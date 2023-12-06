CC=g++
LD=g++

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

# .o files depend on .cpp files with same prefix
# $@ target, $< prerequisite, $* matched stencil
%.o:: %.cpp
	$(CC) $(CPPFLAGS) -c $< -o $@
	$(LD) $(LDFLAGS) $@ -o $*_lib.so

clean:
	rm -f *.o *.so *.pyc
