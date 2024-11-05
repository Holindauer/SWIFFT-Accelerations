#!/bin/bash

make

# Test 1
./test-amx-matmul 
ret=$?
if [ $ret -eq 0 ]; then
    echo && echo "test-amx-matmul... PASS!" && echo
else
    echo "   test-amx-matmul... FAIL!"
fi

# Test 2
./test-amx-fftsum
ret=$?
if [ $ret -eq 0 ]; then
    echo && echo "test-amx-fftsum... PASS!" && echo 
else
    echo "   test-amx-fftsum... FAIL!"
fi


# cleanup
make clean