#!/bin/bash

ratios=( "0.1" "0.15")


for ratio in "${ratios[@]}"
do
  cmd="python main_moco_ratiok.py \
    -a resnet50 \
    --lr 0.03 \
    --batch-size 256 \
    --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 \
     --rank 0 --mlp --moco-t 0.2 --aug-plus --cos --moco-k-ratio ${ratio} --save-folder output/resnet50imgnet${ratio}/ \
    data/imgnet_exp/imgnet${ratio}/ "
  echo ${cmd}
  eval ${cmd}
done



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

