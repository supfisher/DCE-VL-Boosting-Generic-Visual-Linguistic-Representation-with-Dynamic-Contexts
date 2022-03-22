module load nccl/2.4.8
module load openmpi
module remove cuda
module remove cudnn
module load gcc
module load cudnn/7.5.0-cuda10.1.105
source activate vl-bert
tmp_path="/ibex/scratch/mag0a/Github/UNITER"
export PYTHONPATH=$PYTHONPATH:$tmp_path

mpirun -np 4 python train_itm.py --config config/EBS-train-itm-flickr-large-16gpu-hn.json