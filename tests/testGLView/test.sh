#!/bin/bash


dirbak=`pwd`
cd ../../cpp
make GLV
mv GLV_lib.so libGLV.so
echo "==== GLView compiled ... go back .. "
cd $dirbak

LFLAGS="-I../../cpp -L../../cpp"

g++ -o testGLV.x test.cpp $LFLAGS -lGLV -lGL -lSDL2


ln -f -s ../../cpp/libGLV.so  ./libGLV.so
./testGLV.x

