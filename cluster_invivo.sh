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
matlab -batch "main("$DATA"); exit"
echo "Denoising done"
source ../activate_invivo.sh
cd ../demix
python main.py $DATA
echo "Demixing done"
