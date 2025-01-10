LD_LIBRARY_PATH=~/local/cuda-10.0/lib64:$LD_LIBRARY_PATH PATH=~/local/cuda-10.0/bin:$PATH
exe=./Release/ssai-3dstencil
names="7pt 13pt poisson"
for name in $names; do
	$exe $name double
done

for name in $names; do
	$exe $name float
done

