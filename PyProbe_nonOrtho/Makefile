CC=g++
CPPFLAGS=-std=c++11 -O3 -ffast-math -ftree-vectorize 
CPPLFLAGS= $(CPPFLAGS) -shared -Wl,-soname

all: PP


PP:
	$(CC) $(CPPFLAGS) -c -fPIC ProbeParticle.cpp 
	$(CC) $(CPPLFLAGS),ProbeParticle_lib.so -o ProbeParticle_lib.so ProbeParticle.o

clean:
	rm *.o *.so
	
