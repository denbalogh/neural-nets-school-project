#!/bin/bash

cd src

echo "#################"
echo "    COMPILING    "
echo "#################"

make clean
make 

echo "#################"
echo "     RUNNING     "
echo "#################"

nice -n 19 ./network aura
