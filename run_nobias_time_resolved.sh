#!/bin/bash

mkdir -p logs

#for m in 18 60 210 756 2772; do
#for m in 2772; do
#for rate in 0.2 0.3 0.4; do
for rate in 0.5; do
    for m in 756; do
	for i in {11..20}; do
        echo "Running m=$m, itime=$i"

        nohup python -m src.app.main --m $m --readmode --itime $i --mode nobias --rate  $rate\
               > logs/m${m}_itime${i}_rate${rate}_timeresolved.log 2>&1 &
	done
    done
done
