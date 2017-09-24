#!/bin/bash

now=`date +%Y-%m-%d.%H:%M:%S`
mkdir $now
mv *.chtc_log *.chtc_err *.chtc_out $now 2>/dev/null
mv *.pyphi_log *.pkl $now 2>/dev/null
rm pyphi.log
rm -r __pyphi_cache__
