source .env && sbatch --account="$SLURM_ACCOUNT" jobs/train_stage4_from_D512.sh

srun --jobid=55020827 --overlap --pty bash

squeue -u $USER -o "%.18i %.9P %j %.8u %.2t %.10M %.6D %R"