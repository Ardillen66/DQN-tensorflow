#!/bin/bash -l
# number of nodes and cores 
#PBS -l nodes=1:ppn=1
# memory requirements change this when using more/less experience replay samples
#PBS -l mem=16gb
# max run time
#PBS -l walltime=1:00:00
# output and error files
#PBS -o dqn-test.out
#PBS -e dqn-test.err
#PBS -N dqn-test
#PBS -V

module add openblas
cd $HOME
source .bashrc
source activate dqn
cd DQN-tensorflow
python main.py --use_gpu 0 --env_name=CartPole-v0  is_Train=True