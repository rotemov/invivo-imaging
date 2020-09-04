#!/bin/bash
#SBATCH -J II
#SBATCH -o logs/II_%j.out
#SBATCH -N 4
#SBATCH -c 32
#SBATCH --threads-per-core=1
#SBATCH --mem=128G
#SBATCH --mail-type=END
#SBATCH --mail-user=rotem.ovadia@mail.huji.ac.il
##SBATCH -e logs/II_%j.err


DATA=${1:-"/ems/elsc-labs/adam-y/rotem.ovadia/Programs/invivo-imaging/Data/two_cells/cell1"}
FN=${2:-"cell1.bin"}
NORMCORRE=${3:-"YES"}
DETREND=${4:-"YES"}
MOCO=${5:-"YES"}
DEMIX=${6:-"YES"}
CUTOFF_POINT=${7:-"0.9"}
CORR_TH_FIX=${8:-'0.3'}
PATCH_SIZE=${9:-'175'}
BG_RANK=${10:-'4'}
TRUNC_START=${11:-'1000'}
TRUNC_LENGTH=${12:-'5000'}
OUTPUT=${13:-$DATA"/output"}
MOV_IN=${14:-"movReg.tif"}
DETR_SPACING=${15:-'5000'}
ROW_BLOCKS=${16:-'4'}
COL_BLOCKS=${17:-'2'}
STIM_DIR=${18:-""}


echo "Paramaters: "
echo "Data directory: "$DATA
echo "File name: "$FN
echo "NormCoRRe: "$NORMCORRE
echo "Detrending: "$DETREND
echo "Motion correction: "$MOCO
echo "Demixing: "$DEMIX
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

for var in "$@"
do
  var=""
done


PIPELINE_DIR="/ems/elsc-labs/adam-y/rotem.ovadia/Programs/invivo-imaging"
cd $PIPELINE_DIR
source activate_invivo.sh

echo "Starting denoising stage"
cd denoise
if [ $NORMCORRE == "YES" ]
then
  echo "Starting registration"
  matlab -batch "main_bash('"$DATA"','"$FN"'); exit"
  echo "Registration done"
fi
if [ $DETREND == "YES" ]
then
  echo "Starting detrending"
  python denoise.py "$DATA" "$MOV_IN" "$OUTPUT" "$DETR_SPACING" "$ROW_BLOCKS" "$COL_BLOCKS" "$TRUNC_START" "$TRUNC_LENGTH" "$STIM_DIR"
  echo "Detrending finished"
fi
if [ $MOCO == "YES" ]
then
  echo "Starting motion_correction"
  matlab -batch "motion_correction('"$DATA"','"$OUTPUT"'); exit"
  echo "motion_correction finished."
fi
echo "Denoising stage done"


cd ../demix
if [ $DEMIX == "YES" ]
then
  echo "Starting demixing stage"
  python main.py "$DATA" "$CUTOFF_POINT" "$CORR_TH_FIX" "$PATCH_SIZE" "$BG_RANK" "$TRUNC_START" "$TRUNC_LENGTH"
  echo "Demixing stage done"
fi