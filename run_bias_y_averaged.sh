#!/bin/bash

mkdir -p logs

#for m in 18 60 210 2772 10296; do
for m in 756; do
# for m in 2772; do
    for i in 0.3 0.4; do
        echo "Running m=$m, rate=$i"

        nohup python -m src.app.main --m "$m" --batch --mode bias_y --rate "$i" \
            > "logs/m${m}_bias_y_rate${i}.log" 2>&1 &
    done
done
