#!/bin/bash
function tm(){
        echo `date +%s`
}

function diff_tm(){
        TIME_A=$1
        TIME_B=$2
        PT=`expr ${TIME_B} - ${TIME_A}`
        H=`expr ${PT} / 3600`
        PT=`expr ${PT} % 3600`
        M=`expr ${PT} / 60`
        S=`expr ${PT} % 60`
        echo "${H}:${M}:${S}"
}

function get_gpu(){
	nvidia-smi | cat | head -n 8 | tail -n 1 | awk '{print $4}' | awk -F '-' '{print $1}'
}


exe() {
	echo "\$ $@"; "$@";
}

function set_double(){
	#if [ -e ./setting.h ]; then sed -i -e 's/float/double/g' ./setting.h;
	if [ -e ./setting.h ]; then sed -i -e 's/use_dtype 0/use_dtype 1/g' ./setting.h;
	else echo "error, no setting.h";
	fi
}

function set_float32(){
	#if [ -e ./setting.h ]; then sed -i -e 's/double/float/g' ./setting.h;
	if [ -e ./setting.h ]; then sed -i -e 's/use_dtype 1/use_dtype 0/g' ./setting.h;
	else echo "error, no setting.h";
	fi
}
