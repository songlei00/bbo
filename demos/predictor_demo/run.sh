#!/bin/bash

predictors=(
    gp
    kumar_gp
    mlp_gp
)

datasets=(
    autompg
    # concreteslump
    # energy
    # forest
    # parkinsons
    # pumadyn32nm
    # solar
    # stock
    # yacht
    # airfoil
    # autos
    # breastcancer
    # concrete
    # fertility
    # gas
    # housing
    # machine
    # pendulum
    # servo
    # skillcraft
    # sml
    # wine
)

available_gpu=(0)
num_gpu=${#available_gpu[@]}
max_proc=6
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
for predictor in ${predictors[@]}
do
    for dataset in ${datasets[@]}
    do
        read -u7 proc_id
        curr_gpu=${available_gpu[${curr_idx}]}
        curr_idx=$((( $curr_idx + 1 ) % $num_gpu))
        echo 'proc id: '$proc_id', predictor: '$predictor', dataset: '$dataset', gpu idx: '$curr_gpu
        {
            python main.py \
                --predictor=$predictor \
                --dataset=$dataset \
                --optimizer=adam \
                --epochs=200 \
                --device=cuda:$curr_gpu

            sleep 1
            echo >&7 $proc_id
        } &
    done
done

wait
exec 7>&-