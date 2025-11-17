#!/bin/bash
#
#SBATCH --job-name=imp_cho
#SBATCH --output=out_cho.txt
#SBATCH --error=err_cho.txt
#SBATCH -A bsavoie
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=14
#SBATCH --mem=80G
#SBATCH -t 3:00:00
#SBATCH --qos=preemptible

echo Running on host `hostname`
echo Time is `date`
module load conda
source activate /depot/bsavoie/data/Schofield/py38 #/depot/bsavoie/apps/anaconda3/envs/yarp/

python Train_CHO.py config_CHO.json
