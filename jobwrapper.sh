#!/bin/bash
# This script sets up the Python execution environment on its CHTC execute node. 

tar -xzf Python-3.6.2-built.tar.gz
export PATH=$(pwd)/python/bin:$PATH
mkdir home
export HOME=$(pwd)/home

python job.py
