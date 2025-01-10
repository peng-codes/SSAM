#!/bin/bash

source func.sh

export LD_LIBRARY_PATH=~/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
export PATH=~/local/cuda-10.0/bin:$PATH
###########################
ROOT=../

function prof_ppcg(){
	pushd $ROOT
		ppcgs="$(ls -d ppcg-*)"
		#ppcgs=$(ls -d ppcg-j3d7pt)
		GPU=$(get_gpu)

		echo $ppcgs
		for s in $ppcgs; do
			pushd $s
			echo "----------------------------------------"
				#pwd
				make CONFIG=Release clean
				make CONFIG=Release all -j16
				rm -f nvcc-results
				./run.sh > log.tmp
				dtype=$(cat log.tmp | grep "dtype" | awk -F "=" '{print $2}')			
				echo "GPU=" ${GPU}", dtype="${dtype}
				cat log.tmp
				fname=${GPU}.${dtype}.nvcc.results
				rm -f nvcc.results $fname
					
				./prof.sh	
				mv nvcc.results $fname
				cat $fname
			popd
		done
	popd
}

function prof_double(){
	echo "###########Test ppcg double"
	set_double
	prof_ppcg
}

function prof_float(){
	echo "###########Test ppcg float"
	set_float32
	prof_ppcg
}

prof_float
prof_double
