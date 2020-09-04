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

DATA=${1:-"/ems/elsc-labs/adam-y/rotem.ovadia/Programs/invivo-imaging/Data/two_cells/cell1"}
FN=${2:-"cell1.bin"}
CUTOFF_POINT=${3:-"0.9"}
CORR_TH_FIX=${4:-'0.3'}
PATCH_SIZE=${5:-'175'}
BG_RANK=${6:-'4'}
TRUNC_START=${7:-'1000'}
TRUNC_LENGTH=${8:-'5000'}
OUTPUT=${9:-$DATA"/output"}
MOV_IN=${10:-"movReg.tif"}
DETR_SPACING=${11:-'5000'}
ROW_BLOCKS=${12:-'4'}
COL_BLOCKS=${13:-'2'}
STIM_DIR=${14:-""}


echo "Paramaters: "
echo "Data directory: "$DATA
echo "File name: "$FN
echo "Cutoff point: "$CUTOFF_POINT
echo "Correlation threshold fix: "$CORR_TH_FIX
echo "Patch size: "$PATCH_SIZE
echo "Background rank: "$BG_RANK
echo "Truncation start: "$TRUNC_START
echo "Truncation length: "$TRUNC_LENGTH
echo "Output directory: "$OUTPUT
echo "Registered movie: "$MOV_IN
echo "Detrending spacing: "$DETR_SPACING
echo "Row blocks: "$ROW_BLOCKS
echo "COL_BLOCKS: "$COL_BLOCKS
echo "STIM_DIR: "$STIM_DIR



PIPELINE_DIR="/ems/elsc-labs/adam-y/rotem.ovadia/Programs/invivo-imaging"
cd denoise
matlab -batch "main_bash('"$DATA"','"$FN"'); exit"
source $PIPELINE_DIR/activate_invivo.sh
echo "Starting denoise.py"
python denoise.py "$DATA" "$MOV_IN" "$OUTPUT" "$DETR_SPACING" "$ROW_BLOCKS" "$COL_BLOCKS" "$TRUNC_START" "$TRUNC_LENGTH" "$STIM_DIR"
matlab -batch "motion_correction('"$DATA"','"$OUTPUT"'); exit"
echo "Denoising done"
cd ../demix
python main.py "$DATA" "$CUTOFF_POINT" "$CORR_TH_FIX" "$PATCH_SIZE" "$BG_RANK" "$TRUNC_START" "$TRUNC_LENGTH"
echo "Demixing done"

