#!/bin/bash
#SBATCH -J II
#SBATCH -o II.out
#SBATCH -e II.err
#SBATCH -N 5
#SBATCH -c 60
#SBATCH --mem=32G
#SBATCH --mail-type=END
#SBATCH --mail-user=rotem.ovadia@mail.huji.ac.il
#SBATCH --export=DATA="/ems/elsc-labs/adam-y/rotem.ovadia/Programs/invivo-imaging/Data/demo_data"

cd denoise
# matlab2017b -batch "main; exit"
matlab -batch "main("$DATA"); exit"
echo "main('"$DATA"'); exit"
# matlab2017b -nodisplay -nosplash -r "main; exit"
echo "Denoising done"


eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate invivo
echo "Demix invivo env activated"
cd ../demix
python main.py $DATA
echo "Demixing done"
