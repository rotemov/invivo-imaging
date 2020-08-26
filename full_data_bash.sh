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
#SBATCH --export=DATA="/ems/elsc-labs/adam-y/rotem.ovadia/Programs/invivo-imaging/Data/demo_data",FN="raw_data.tif",MOV_IN="movReg.tif",DETR_SPACING=5000,ROW_BLOCKS=4,COL_BLOCKS=2,STIM_DIR="",TRUNC_START=1,TRUNC_LENGTH=5000
PIPELINE_DIR="/ems/elsc-labs/adam-y/rotem.ovadia/Programs/invivo-imaging"
OUTPUT=$DATA"/output"

cd denoise
matlab -batch "main_bash('"$DATA"','"$FN"'); exit"
source $PIPELINE_DIR/activate_invivo.sh
echo "Starting denoise.py"
python denoise.py $DATA $MOV_IN $OUTPUT $DETR_SPACING $ROW_BLOCKS $COL_BLOCKS $TRUNC_START $TRUNC_LENGTH $STIM_DIR
matlab -batch "motion_correction('"$DATA"','"$OUTPUT"'); exit"
echo "Denoising done"
cd ../demix
python main.py $DATA
echo "Demixing done"

