#!/bin/bash
#SBATCH -J invivo_imaging_test
#SBATCH -o II.out
#SBATCH -e II.err
#SBATCH -N 4
#SBATCH -c 40
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rotem.ovadia@mail.huji.ac.il

cd denoise
# matlab2017b -batch "main; exit"
matlab -batch "main; exit"
# matlab2017b -nodisplay -nosplash -r "main; exit"
echo "Denoising done"


eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate invivo
echo "invivo env activated"
cd ../demix
python main.py
echo "Demixing done"
