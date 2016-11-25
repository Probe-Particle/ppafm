CC=g++
LD=g++
CPPFLAGS= -std=c++11 -O3 -ffast-math -ftree-vectorize 
LDFLAGS= -shared

%::%.cpp
	$(CC) -c $< -o $@.o $(CFLAGS)
	$(LD) $(LDFLAGS) $@.o -o $@_lib.so

PP: ProbeParticle

GU: GridUtils

MP: Multipoles

all: PP GU MP

clean:
	rm *.o *.so *.pyc
