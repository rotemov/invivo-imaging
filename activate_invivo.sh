#!/bin/bash
. ${HOME}/intel/parallel_studio_xe_2020.2.108/bin/psxevars.sh
. ${HOME}/.bashrc
echo "bashrc sourced"
. ${HOME}/Programs/anaconda3/bin/activate invivo
echo "invivo env activated"
# OpenBLAS
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${HOME}/Programs/OpenBLAS"
