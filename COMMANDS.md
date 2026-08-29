source .env && sbatch --account="$SLURM_ACCOUNT" jobs/train_stage4_from_D512.sh

srun --jobid=XXXXXX --overlap --pty bash