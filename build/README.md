Many shared libraries needed by Python modules are missing from the execute nodes. 
It is easiest to just take them with you wherever you go. 
Profiling to find the minimal set of necessary libraries seems is hard and has issues. 
Instead, just take the entire /usr/lib64/ with you. 

First, copy the directory tree while translating any symlinks TO COPIED FILES wherever possible.
Because 90% of lib64 is symlinks, this is necessary to avoid a massive copy.
```
mkdir SS
while read F; do
  cp -av /usr/lib64/$F SS/
done < shared_libraries.txt
```
Now, get the shared libraries from outside /usr/lib64/.
This could be done more intelligently, but there are so few of these files...
```
while read F; do
  cp --remove-destination /usr/lib64/$F SS/
done < shared_libraries_other.txt
```
Good to go!
