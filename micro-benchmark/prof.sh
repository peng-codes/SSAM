#!/bin/bash

function get_gpu(){
        nvidia-smi | cat | head -n 8 | tail -n 1 | awk '{print $4}' | awk -F '-' '{print $1}'
}

export CUDA_INSTALL_PATH=$LOCAL_HOME/cuda-10.0
export LD_LIBRARY_PATH=$LOCAL_HOME/cuda-10.0/lib64:$LD_LIBRARY_PATH

name=$(get_gpu).results

rm -f $name

# make global
make pipeline
make shared

# echo "Latency of Global memory"
# ./bin/linux/release/global >> $name 


echo "Latency evaluation : +-*/ shfl*" >>$name
./bin/linux/release/pipeline >> $name 


echo "shared memory access latency" >> $name
 ./bin/linux/release/shared >> $name


cat $name







