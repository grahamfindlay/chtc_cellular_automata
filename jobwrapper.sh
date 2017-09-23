#!/bin/bash
# This script sets up the Python execution environment on its CHTC execute node. 

tar -xzf Python-3.6.2-built.tar.gz
export PATH=$(pwd)/python/bin:$PATH
mkdir home
export HOME=$(pwd)/home
tar -xzf SLIBS.tar.gz
export LD_LIBRARY_PATH=$(pwd)/SS

python job.py $1 $2 $3
