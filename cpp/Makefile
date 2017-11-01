CC=g++-5
LD=g++-5
CPPFLAGS= -std=c++11 -O3 -ffast-math -ftree-vectorize -fPIC
LDFLAGS= -shared

all: PP GU MP

PP: ProbeParticle.o

GU: GridUtils.o

MP: Multipoles.o

# .o files depend on .cpp files with same prefix
# $@ target, $< prerequisite, $* matched stencil
%.o:: %.cpp
	$(CC) $(CPPFLAGS) -c $< -o $@
	$(LD) $(LDFLAGS) $@ -o $*_lib.so

clean:
	rm -f *.o *.so *.pyc
