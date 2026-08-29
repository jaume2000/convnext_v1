source .env && sbatch --account="$SLURM_ACCOUNT" jobs/train_stage4_from_D512.sh

srun --jobid=55020827 --overlap --pty bash