# !/usr/bin/bash

function get_gpu(){
	nvidia-smi | cat | head -n 8 | tail -n 1 | awk '{print $4}' | awk -F '-' '{print $1}' 
}

dtype=float
if [ $1 = double ]; then
	dtype=double
fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/local/cuda-10.0/lib64

rm -f log.conv2D.csv

imgSize=8192
for ((fsize=2; fsize<22; fsize++)); do
	cmd="./Release/ssai-2dconv $dtype $imgSize $fsize"
	echo "cmd="$cmd
	eval $cmd
done

dst_name=$(get_gpu).${dtype}.log.conv2D.csv
echo "result is "$dst_name
mv log.conv2D.csv $dst_name
