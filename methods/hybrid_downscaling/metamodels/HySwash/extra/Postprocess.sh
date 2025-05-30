#!/bin/bash
#SBATCH --job-name=Vsin_veg
#SBATCH --output=output_%A_%a.err
#SBATCH --error=error_%A_%a.err
#SBATCH --partition=geocean_priority
##SBATCH --nodelist=geocean01


cd /home/grupos/geocean/valvanuz/HySwash/BlueMath/methods/hybrid_downscaling/metamodels/HySwash/
python utils/postprocess_parallel.py $SLURM_ARRAY_TASK_ID 
