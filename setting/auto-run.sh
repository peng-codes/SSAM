#!/usr/bin/bash
source ./func.sh

CUR_DIR=$(pwd)
SERVERS="warsaw rwbc-v100"
echo $CUR_DIR

for s in $SERVERS; do
	start=$(tm)
	cmd="ssh -t $s 'cd ${CUR_DIR} && echo $(pwd) && ./test-all.sh'"
	echo $cmd
	eval $cmd
	end=$(tm)
	dif=$(diff_tm $start $end)
	echo "$cmd, start=$start, end=$end, total_time=$dif" >> time.txt
done




