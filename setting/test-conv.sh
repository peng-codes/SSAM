# !/bin/bash

export LD_LIBRARY_PATH=~/projects/halide-bin/bin:$LOCAL_HOME/gcc-5.4/lib64:$LOCAL_HOME/cuda-10.0/lib64:$LD_LIBRARY_PATH
export PATH=$LOCAL_HOME/gcc-5.4/bin:$LOCAL_HOME/cuda-10.0/bin:$PATH

abspath() {
        python -c "import os,sys; print os.path.realpath(sys.argv[1])" $1
}

ROOT=$(abspath ../)
LogFile=$(pwd)/log.txt
echo "ROOT=$ROOT"
echo "LogFile=$LogFile"

#2dconv
function test_ssai_2dconv(){
	echo ${FUNCNAME[ 0 ]}>>$LogFile
	pushd $ROOT/ssai-2dconv
	make -j8
	./run.sh
	popd
}


#halide_2dconv
function test_halide_2dconv(){
	echo ${FUNCNAME[ 0 ]}>>$LogFile
	pushd $ROOT/halide-2dconv
	make build
	./prof.sh
	popd
}

#cuFFT
function test_cufft_2dconv(){
	echo ${FUNCNAME[ 0 ]}>>$LogFile
	pushd $ROOT/cufft-2dconv
	make 
	./run.sh
	popd
}

#npp
function test_npp_2dconv(){
	echo ${FUNCNAME[ 0 ]}>>$LogFile
	pushd $ROOT/npp-2dconv
	make 
	./run.sh
	popd
}

#cuDNN
function test_cuDNN_2dconv(){
	echo ${FUNCNAME[ 0 ]}>>$LogFile
	pushd $ROOT/cudnn-conv2d
	make 
	./run.sh
	popd
}

#arrayfire
function test_arrayfire(){
	echo ${FUNCNAME[ 0 ]}>>$LogFile
	pushd $ROOT/arrayfire
	make download
	make build
	make test
	popd
}

# test_halide_2dconv

# test_arrayfire
# test_cuDNN_2dconv
# test_npp_2dconv
# test_cufft_2dconv
# test_halide_2dconv
test_ssai_2dconv
