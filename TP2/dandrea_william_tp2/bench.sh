#!/bin/bash

rm -f a.out res.txt

echo "[INFO] Usage: bench.sh SOURCE_FILE N_THREADS USE_CPUS"

if [[ $# -lt 2 ]]
then
    N_THREADS=1000
else
    N_THREADS=$2
fi

if [[ $# -lt 3 ]]
then
    USE_CPUS=1
else
    USE_CPUS=$3
fi
echo $N_THREADS $USE_CPUS
gcc $1 -g -Wall -O3 -msse3 -ftree-vectorize -ftree-vectorizer-verbose=0 \
    -finline-functions -lpthread -march=native -mtune=native \
    -DUSE_CPUS=$USE_CPUS -DN_THREADS=$N_THREADS -DUSE_CPUS=$USE_CPUS \
    -D_GNU_SOURCE -pipe -lm

for a in {1..3};do
    ./a.out 
done

rm -f a.out res.txt

