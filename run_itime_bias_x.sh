#!/bin/bash

mkdir -p logs

#for m in 18 60 210 756 2772; do
for m in 2772; do
    for i in {11..20}; do
        echo "Running m=$m, itime=$i"

        python -m src.app.main --m $m --readmode --mode bias_x --itime $i \
            > logs/m${m}_itime${i}.log 2>&1
    done
done
