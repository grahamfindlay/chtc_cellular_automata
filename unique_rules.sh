#!/bin/bash

SUBMIT_FILE="unique_rules.sub"
N_CELLS=5

UNIQUE_RULES=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 18 19 22 23 24 25 26 27 28 29 30 32 33 34 35 36 37 38 40 41 42 43 44 45 46 50 51 54 56 57 58 60 62 72 73 74 76 77 78 90 94 104 105 106 108 110 122 126 128 130 132 134 136 138 140 142 146 150 152 154 156 160 162 164 168 170 172 178 184 200 204 232)

echo "queue rule, n, z from (" >> $SUBMIT_FILE
for rule in ${UNIQUE_RULES[@]}; do
  for ((z = 0; z <= $N_CELLS; z++)); do
    echo "$rule, $N_CELLS, $z" >> $SUBMIT_FILE
  done
done
echo ")" >> $SUBMIT_FILE


