# !/bin/bash

function get_gpu(){
	nvidia-smi | cat | head -n 8 | tail -n 1 | awk '{print $4}' | awk -F '-' '{print $1}'
}

pushd ../Rawat-reg-opt
	(./run.sh)
	(
		name=$(get_gpu).nvcc.results
		rm -f $name
		./get_results.sh >> $name 
	)
popd
