#!/bin/bash
#SBATCH -J II
#SBATCH -o logs/II_%j.out
#SBATCH -e logs/II_%j.err
#SBATCH -N 2
#SBATCH -c 20
#SBATCH --mem=32G
#SBATCH --mail-type=END
#SBATCH --mail-user=rotem.ovadia@mail.huji.ac.il
#SBATCH --export=DATA="/ems/elsc-labs/adam-y/rotem.ovadia/Programs/invivo-imaging/Data/two_cells",FN="cell1.bin"

cd denoise
matlab -batch "main('"$DATA","$FN"'); exit"
echo "Denoising done"
source ../activate_invivo.sh
cd ../demix
python main.py $DATA
echo "Demixing done"
