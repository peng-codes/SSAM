#!/bin/bash

source func.sh

ROOT=..

##########################################
echo "############# warning #############"
pushd $ROOT
	echo $(pwd)
	dirs=$(ls -d */)
	for s in $dirs; do
		echo "s=$s, pwd=$(pwd)"
		(
			cd $s
			echo "pwd=$(pwd)"
			for i in P100.* V100.* *.results *.results *.tmp *.csv; do
				if [ -e $i ]; then rm -f $i; fi
			done

			if [ -e prof.sh ]; then 
				sed -i -e 's/nvprof --print-gpu-trace/nvprof --normalized-time-unit ms --print-gpu-trace/g' prof.sh
			fi		
		)
    done
	files=$(find . -type f -name "P100.*" -o -name "V100.*")
	for s in $files; do
		rm -f $s
	done
popd
