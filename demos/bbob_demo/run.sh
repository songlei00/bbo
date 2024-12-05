#!/bin/bash

algos=(
    random
    ls
    re
    pso
    random_bo
)

funcs=(
    GriewankRosenbrock
    Lunacek
    Rastrigin
    RosenbrockRotated
    SharpRidge
    AttractiveSector
    BentCigar
    DifferentPowers
    Discus
    Gallagher101Me
    Gallagher21Me
    Katsuura
    LinearSlope
    NegativeMinDifference
    NegativeSphere
    SchaffersF7IllConditioned
    SchaffersF7
    StepEllipsoidal
    Weierstrass
)
min_seed=0
max_seed=4

available_gpu=(0)
num_gpu=${#available_gpu[@]}
max_proc=20
echo 'GPU idx: '${available_gpu[@]}
echo 'Number of GPU: '$num_gpu
echo 'Max number of processes: '$max_proc

fifo_name="/tmp/$$.fifo"
mkfifo $fifo_name
exec 7<>${fifo_name}
# rm $fifo_name

for ((i=0; i<$max_proc; i++))
do
    echo $i
done >&7

curr_idx=0
for ((seed=${min_seed}; seed<=${max_seed}; seed++))
do
    for algo in ${algos[@]}
    do
        for func in ${funcs[@]}
        do
            read -u7 proc_id
            curr_gpu=${available_gpu[${curr_idx}]}
            curr_idx=$((( $curr_idx + 1 ) % $num_gpu))
            echo 'proc id: '$proc_id', algo: '$algo', func: '$func', gpu idx: '$curr_gpu
            {
                python main_bbob.py \
                    --algo=$algo \
                    --func=$func \
                    --seed=$seed

                sleep 1
                echo >&7 $proc_id
            } &
        done
    done
done

wait
exec 7>&-