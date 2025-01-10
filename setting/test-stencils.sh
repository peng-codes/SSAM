#!/bin/bash

export LD_LIBRARY_PATH=~/projects/halide-bin/bin:$LOCAL_HOME/cuda-10.0/lib64:$LOCAL_HOME/gcc-5.4/lib64:$LD_LIBRARY_PATH
export PATH=$LOCAL_HOME/cuda-10.0/bin:$LOCAL_HOME/gcc-5.4/bin:$PATH


abspath() {
        python -c "import os,sys; print os.path.realpath(sys.argv[1])" $1
}

ROOT=$(abspath ../)
LogFile=$(pwd)/log.txt
echo "ROOT=$ROOT"
echo "LogFile=$LogFile"

#####
dirs=$(ls -d $ROOT/setting $ROOT/ssai-* $ROOT/ppcg-* $ROOT/halid*)
for s in $dirs; do
	#pushd $s
	chmod +x ${s}/*.sh && sed -i 's/\r$//' ${s}/*.sh
	#sed -i -e 's/nvprof --print-gpu-trace/nvprof --print-gpu-trace/g' prof.sh
	#popd
done

#### halide
function prof_halide(){
	echo ${FUNCNAME[ 0 ]}>>$LogFile
	halide="$ROOT/halide-2dstencil $ROOT/halide-3dstencil"
	for s in $halide; do
		pushd ${s}
			make clean all
			./prof.sh
		popd
	done
}
#### 2d stencil
function prof_2dstencil(){
	echo ${FUNCNAME[ 0 ]}>>$LogFile
	pushd $ROOT/ssai-2dstencil
		make clean all
		./prof.sh
	popd
}

####3d stencil
function prof_3dstencil(){
	echo ${FUNCNAME[ 0 ]}>>$LogFile
	pushd $ROOT/ssai-3dstencil
		make clean all
		./prof.sh
	popd
}

#### ppcg
function prof_ppcg(){
	echo ${FUNCNAME[ 0 ]}>>$LogFile
	pushd $ROOT/setting
		make clean all 
		./test-ppcg.sh
	popd
}
#### stencilGen
function prof_stencilgen(){
	echo ${FUNCNAME[ 0 ]}>>$LogFile
	pushd $ROOT/stencilGen
		make clean all
		./get-all-time.sh
	popd
}

# pre-head GPUs
#prof_3dstencil

#prof_halide
prof_2dstencil
prof_3dstencil
#prof_ppcg
#prof_stencilgen









