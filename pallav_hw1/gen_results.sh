#!/bin/bash

CMD="./matsum"
CMD_AVX="./matsum_avx"
OUTPUT="./results"

if [ ! -d $OUTPUT ]
then
	mkdir -p $OUTPUT
fi

SEQ_OUTPUT_FILE="$OUTPUT/seq_compute_time.csv"
if [ -f $SEQ_OUTPUT_FILE ]
then
	rm -f $SEQ_OUTPUT_FILE
fi

AVX_OUTPUT_FILE="$OUTPUT/avx_compute_time.csv"
if [ -f $AVX_OUTPUT_FILE ]
then
	rm -f $AVX_OUTPUT_FILE
fi

if [ -f $CMD ] || [ -f $CMD_AVX ]
then
	make clean
fi
make all

SIZES='1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 
262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728' 


for size in $SIZES
do
	echo "Running Sequential"
	time=`$CMD $size | grep "Compute Time" | awk -F ":" '{print $2}'| awk '{print $1}'`
	echo "$size:$time"| tee -a $SEQ_OUTPUT_FILE
	echo "Running with AVX"
	time=`$CMD_AVX $size | grep "Compute Time" | awk -F ":" '{print $2}'| awk '{print $1}'`
	echo "$size: $time" | tee -a $AVX_OUTPUT_FILE
done
