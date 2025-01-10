#!/bin/bash
export LD_LIBRARY_PATH=~/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
export PATH=~/local/cuda-10.0/bin:$PATH 

function get_gpu(){
	nvidia-smi | cat | head -n 8 | tail -n 1 | awk '{print $4}' | awk -F '-' '{print $1}' 
}
###########################################
name="7pt 13pt poisson"

rm -f nvcc.results
GPU=$(get_gpu)
echo "GPU="${GPU}
rm -f ${GPU}.float32.nvcc.results ${GPU}.double.nvcc.results
exe=./Release/ssai-3dstencil
for s in $name; do
	echo "prof : " $s, double, float
	$exe $s double
	$exe $s float
	nvprof --normalized-time-unit ms --print-gpu-trace $exe $s double > /dev/null 2>>${GPU}.double.nvcc.results
	nvprof --normalized-time-unit ms --print-gpu-trace $exe $s float > /dev/null 2>>${GPU}.float32.nvcc.results
done
cat *.nvcc.results | grep 'void' | awk '{print $2, $17 ,$21}'
