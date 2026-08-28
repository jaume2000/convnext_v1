source .env && sbatch --account="$SLURM_ACCOUNT" jobs/shared_convnext_ablation.sh

srun --jobid=XXXXXX --overlap --pty bash