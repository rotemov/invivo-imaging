#!/bin/bash
cd denoise
matlab -batch "main; exit"
echo "Denoising done"


eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate invivo
echo "invivo env activated"
cd ../demix
python main.py
echo "Demixing done"
