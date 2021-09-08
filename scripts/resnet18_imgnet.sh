#!/bin/bash
#PBS -l select=1:vmem=256GB:ncpus=8:ngpus=4
#PBS -l pmem=120GB
#PBS -N prems9_tune
#PBS -j oe
#PBS -o runskd/output_off6.log
#PBS -q research
#PBS -v CONTAINER_IMAGE=nvcr.io/nvidia/pytorch:20.06-py3

echo "CONTAINER_IMAGE=nvcr.io/nvidia/pytorch:20.06-py3"


cd "$PBS_O_WORKDIR" || exit $?
echo ${PBS_O_WORKDIR}



USER=anhdung_dinh # Replace with your own HPC account name
#/home/users/$USER/.conda/envs/yolodso/bin/pip config set global.target /home/users/$USER/.conda/envs/yolodso/bin/
#export PATH=/home/users/$USER/.conda/envs/yolodso/bin/:$PATH
#export PYTHONPATH=/home/users/$USER/.conda/envs/yolodso/lib/python3.8/site-packages/:$PYTHONPATH
python_alias="/home/users/${USER}/.conda/envs/yolodso/bin/python3.8"
nvidia-smi

cmd="${python_alias} main_moco.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 256 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 \
   --rank 0 --mlp --moco-t 0.2 --aug-plus --cos --save-folder output/resnet18imgnet \
  ../imagenet/ "

echo ${cmd}
eval ${cmd}

#weight="runs/coco/exp0_cocotinyv4/weights/best_cocotinyv4_strip.pt"
#weight="''"
#threshold=( "0.2" )
#kdschs=("5")
#name="tkd_p4csp_tiny_prems"
#t_weight="runs/exp9_prems_vocyolov4-csp/weights/best_prems_vocyolov4-csp.pt"
#
#for kdsch in "${kdschs[@]}"
#do
#for th in "${threshold[@]}"
#do
#  cmd="${python_alias} -m torch.distributed.launch --nproc_per_node 4 --master_port=2211\
#   train_kd.py  --batch-size 64 \
#  --img 448 448 \
#  --data voc.yaml \
#  --cfg voc/vocyolov4-tiny-v4.yaml \
#  --weights ${weight} \
#  --t_weights ${t_weight} \
#  --sync-bn \
#  --device 0,1,2,3 \
#  --name ${name}_sch${kdsch}_thr${th} \
#  --kd_str ${kdsch}\
#  --kd_thr ${th} \
#  --epochs 600 \
#  --logdir runskd/vocofficial\
#  --class_loss l2
#  --maxnw 4
#  --resume"
#  echo ${cmd}
#  eval ${cmd}
#done
#done

