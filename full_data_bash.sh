#!/bin/bash
#SBATCH -J II
#SBATCH -o /ems/elsc-labs/adam-y/rotem.ovadia/Programs/invivo-imaging/logs/II_%j.log
#SBATCH -N 4
#SBATCH -c 32
#SBATCH --threads-per-core=1
#SBATCH --mem=64G
#SBATCH --mail-type=END
#SBATCH --mail-user=rotem.ovadia@mail.huji.ac.il

# Parsing input args into variables
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
RESIDUAL_CUT=${18:-"0.6"}
TH_LVL=${19:-"4"}
PASS_NUM=${20:-"1"}
BG_MASK=${21:-"0"}
MERGE_CORR_THR=${22:-"0.8"}
SUP_ONLY=${23:-"0"}
REMOVE_DIMMEST=${24:-"0"}
UPDATE_AC_KEEP_SHAPE=${25:-"1"}
UPDATE_AC_MAX_ITER=${26:-"35"}
UPDATE_AC_TOL=${27:-"1e-8"}
UPDATE_AC_MERGE_OVERLAP_THR=${28:-"0.8"}
BG_REG_MAX_ITER=${29:-"1000"}
BG_REG_LR=${30:-"0.001"}
STIM_DIR=${31:-""}

# Deactivating the CL args to enable sourcing in the script
set --

# Printing parameter for log
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
echo "Column blocks: "$COL_BLOCKS
echo "Stimulation directory: "$STIM_DIR
echo "Threshold level: "$TH_LVL
echo "Number of passes: "$PASS_NUM
echo "Background mask: "$BG_MASK
echo "Merge correlation threshold: "$MERGE_CORR_THR
echo "Quick run: "$SUP_ONLY
echo "Remove dimmest: "$REMOVE_DIMMEST
echo "Residual cut: "$RESIDUAL_CUT
echo "Update AC max iterations: "$UPDATE_AC_MAX_ITER
echo "Update AC tol: "$UPDATE_AC_TOL
echo "Update AC merge overlap threshold: "$UPDATE_AC_MERGE_OVERLAP_THR
echo "Update AC keep shape: "$UPDATE_AC_KEEP_SHAPE
echo "Background regression learning rate: "$BG_REG_LR
echo "Background regression max iterations: "$BG_REG_MAX_ITER

# Activating environment
PIPELINE_DIR="/ems/elsc-labs/adam-y/rotem.ovadia/Programs/invivo-imaging"
cd $PIPELINE_DIR
. ./activate_invivo.sh

# Running the different steps
echo "Starting denoising stage"
cd denoise
if [ $NORMCORRE == "1" ]; then
  echo "Starting registration"
  matlab -batch "main_bash('"$DATA"','"$FN"'); exit"
  echo "Registration done"
else
  echo "Skipping NormCoRRe"
fi
if [ $DETREND == "1" ]; then
  echo "Starting detrending"
  python denoise.py "$DATA" "$MOV_IN" "$OUTPUT" "$DETR_SPACING" "$ROW_BLOCKS" "$COL_BLOCKS" \
  "$TRUNC_START" "$TRUNC_LENGTH" "$STIM_DIR"
  echo "Detrending finished"
else
  echo "Skipping detrending"
fi
if [ $MOCO == "1" ]; then
  echo "Starting motion_correction"
  matlab -batch "motion_correction('"$DATA"','"$OUTPUT"'); exit"
  echo "motion_correction finished."
else
  echo "Skipping motion_correction"
fi
echo "Denoising stage done"


cd ../demix
if [ $DEMIX == "1" ]; then
  echo "Starting demixing stage"
  python main.py "$DATA" "$CUTOFF_POINT" "$CORR_TH_FIX" "$PATCH_SIZE" "$BG_RANK" \
  "$TRUNC_START" "$TRUNC_LENGTH" "$TH_LVL" "$PASS_NUM" "$BG_MASK" "$MERGE_CORR_THR" \
  "$SUP_ONLY" "$REMOVE_DIMMEST" "$RESIDUAL_CUT" "$UPDATE_AC_MAX_ITER" "$UPDATE_AC_TOL" \
  "$UPDATE_AC_MERGE_OVERLAP_THR" "$UPDATE_AC_KEEP_SHAPE" "$BG_REG_LR" "$BG_REG_MAX_ITER"
  echo "Demixing stage done"
fi

# mv $PIPELINE_DIR/logs/* $OUTPUT/.