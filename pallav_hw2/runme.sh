#!/bin/bash

make clean
make all
rm -f ./output.log

for nth in 1 2 4 8 16 32 64 128 256 512 1024
do
	out=$(./divergent ${nth} | grep 'time')
	out1=$(echo ${out} | awk -F = '{print $3}' | awk '{print $1}')
	out2=$(echo ${out} | awk -F = '{print $2}' | awk '{print $1}')
	echo ${nth} ${out2} ${out1} >> output.log
	echo ${out}
done
echo "Output is in output.log"
