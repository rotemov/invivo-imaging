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
#SBATCH --export=DATA="/ems/elsc-labs/adam-y/rotem.ovadia/Programs/invivo-imaging/Data/two_cells/cell2"
#SBATCH --export=FN="cell2.bin"
#SBATCH --export=MOV_IN="movReg.tif"
#SBATCH --export=DETR_SPACING=5000
#SBATCH --export=ROW_BLOCKS=4
#SBATCH --export=COL_BLOCKS=2
#SBATCH --export=STIM_DIR=""
#SBATCH --export=TRUNC_START=1
#SBATCH --export=TRUNC_LENGTH=5000


cd denoise
matlab -batch "main_bash('"$DATA"','"$FN"'); exit"
source ~/Programs/invivo-imaging/activate_invivo.sh
echo "denoise.py $DATA $MOV_IN $DATA"/output" $DETR_SPACING $ROW_BLOCKS $COL_BLOCKS $TRUNC_START $TRUNC_LENGTH $STIM_DIR"
python denoise.py $DATA $MOV_IN $DATA"/output" $DETR_SPACING $ROW_BLOCKS $COL_BLOCKS $TRUNC_START $TRUNC_LENGTH $STIM_DIR
matlab -nojvm -nodisplay -nosplash -r ""home=$DATA;output=$DATA"/output";motion_correction; exit;""
echo "Denoising done"
cd ../demix
python main.py $DATA
echo "Demixing done"

