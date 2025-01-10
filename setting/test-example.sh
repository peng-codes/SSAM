#!/bin/bash

source func.sh

GPU=$(get_gpu)

example_dir=~/projects/gpu-reg-stencil/ppopp-artifact/examples
working=~/projects/gpu-reg-stencil/ppopp-artifact/${GPU}-examples

chmod +x ${example_dir}/*.sh

cp -rT $example_dir $working

pushd ${working}
	./mybenchmark.sh
popd

