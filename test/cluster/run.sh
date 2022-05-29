#!/bin/bash
#SBATCH --nodes=2 # use 4 nodes
#SBATCH --gres=gpu:1 # each node we only use one GPU
#SBATCH --ntasks=2 # there are totally 4 tasks. In this case the helloworld.py will be run 4 times. These 4 tasks will be evenly distributed to 4 nodes.
#SBATCH -o out
#SBATCH --error=err
#SBATCH --mem=6G
#SBATCH --time=00:01:00

<<<<<<< HEAD
=======
module load cuda/10.1
module load openmpi/4.0.1
module load python/3.6.8

>>>>>>> 7dc7b22b8a538babdbd940c8c434e1ff30301d5e
srun python helloworld.py
