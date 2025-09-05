#!/bin/bash
#SBATCH --job-name=Vsin_veg
#SBATCH --output=output_%A_%a.err
#SBATCH --error=error_%A_%a.err
#SBATCH --partition=geocean_priority
##SBATCH --nodelist=geocean01


id=$(printf "%04d\n" "$SLURM_ARRAY_TASK_ID")
folder="/home/grupos/geocean/valvanuz/HySwash/BlueMath/methods/hybrid_downscaling/metamodels/HySwash/output_Veggy_Hs_mono/$id"
launchSwash.sh --case-dir $folder > $folder/wrapper_out.log 2> $folder/wrapper_err.log
