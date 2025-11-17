#!/bin/bash
#
#SBATCH --job-name=imp_flu
#SBATCH --output=out_flu.txt
#SBATCH --error=err_flu.txt
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

python Train_fluorine.py config_fluorine.json
