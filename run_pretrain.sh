#!/bin/bash
#SBATCH --job-name=crate4gpt_pretrain
#SBATCH --output=nohup_pretrain.out
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:A5000:8
#SBATCH --nodes=4
#SBATCH --export=ALL
#SBATCH --no-requeue
#SBATCH --gpu-bind=closest

echo "node list: "$SLURM_JOB_NODELIST
echo "master address: "$MASTER_ADDR

srun --jobid $SLURM_JOBID \
     --export=ALL \
     bash -c 'echo "slurm process id: "$SLURM_PROCID && python \
    -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=4 \
    --node_rank=$SLURM_PROCID \
    --master_port=19500 \
    --master_addr=$MASTER_ADDR \
    --use_env \
    train.py config/train_gpt2.py'
