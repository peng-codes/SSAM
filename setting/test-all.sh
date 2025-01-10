#!/bin/bash

export CUDA_HOME=/usr/local/cuda-10.0

source func.sh

abspath() {
        python -c "import os,sys; print os.path.realpath(sys.argv[1])" $1
}

ROOT=$(abspath ../)
LogFile=$(pwd)/log.txt
echo "ROOT=$ROOT"
echo "LogFile=$LogFile"

rm -rf $LogFile

################################################
echo "############# warning pre-run#############"

pushd $ROOT
	# echo $(pwd)
	dirs=$(ls -d */)
	for s in $dirs; do
		echo "s=$s, pwd=$(pwd)"
		(
			cd $s
			# echo "pwd=$(pwd)"
			chmod +x *.sh && sed -i 's/\r//g' *.sh
			if [ -e prof.sh ]; then 
				sed -i -e 's/nvprof --print-gpu-trace/nvprof --normalized-time-unit ms --print-gpu-trace/g' prof.sh
			fi		
		)
    done
popd

# test Rawat-reg-opt
# (./test-Rawat-reg-opt.sh)

# test micro-benchmark
(./test-micro-benchmark.sh)

# test all stencils
(./test-stencils.sh)

# test convolution
(./test-conv.sh)

