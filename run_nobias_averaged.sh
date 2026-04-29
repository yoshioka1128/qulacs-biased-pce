#!/bin/bash

mkdir -p logs

#for m in 18 60 210 756 2772; do
for m in 756; do
# for m in 2772; do
    for i in 0.3 0.4 0.5; do
        echo "Running m=$m, rate=$i"

        nohup python -m src.app.main --m "$m" --batch --mode nobias --rate "$i" \
            > "logs/m${m}_nobias_rate${i}.log" 2>&1 &
    done
done
