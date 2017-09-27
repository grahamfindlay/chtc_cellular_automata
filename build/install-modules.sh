#!/bin/bash

# The following packages must be installed on the CHTC build node where Python is built.
# Assumes you followed the directions at http://chtc.cs.wisc.edu/python-jobs.shtml.
export PATH=$(pwd)/python/bin:$PATH
pip3 install "git+https://github.com/grahamfindlay/pyphi.git@a2e846adb16f064693a805a75a517b18e1a638a7"
