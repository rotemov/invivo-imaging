#!/bin/bash
#SBATCH -J II
#SBATCH -o logs/II_%j.out
#SBATCH -e logs/II_%j.err
#SBATCH -N 4
#SBATCH -c 32
#SBATCH --threads-per-core=1
#SBATCH --mem=128G
#SBATCH --mail-type=END
#SBATCH --mail-user=rotem.ovadia@mail.huji.ac.il
#SBATCH --export=DATA="/ems/elsc-labs/adam-y/rotem.ovadia/Programs/invivo-imaging/Data/two_cells/cell2",FN="cell2.bin"

cd denoise
matlab -batch "main('"$DATA"','"$FN"'); exit"
echo "Denoising done"
source ../activate_invivo.sh
cd ../demix
python main.py $DATA
echo "Demixing done"

