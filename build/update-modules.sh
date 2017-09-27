#!/bin/bash

# Don't forget to untar your previously-built python first!
export PATH=$(pwd)/python/bin:$PATH
# Pip points to the location of the python executable at the time it was installed. 
# This path depends on the interactive CHTC node, and changes from job to job.
# Therefore, update pip with the new location, by replacing the first line of the pip file. 
sed -i "1c\#!`which python3.6`" `which pip3`
pip3 uninstall pyphi
pip3 install "git+https://github.com/grahamfindlay/pyphi.git@ac77e5758feeb6299dbae54c1a475d7ae209ac9b"
# Now re-tar your updated installation!
