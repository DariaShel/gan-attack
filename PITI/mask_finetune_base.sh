export CUDA_LAUNCH_BLOCKING=1
export LOGDIR=/home/d-d-sh/adv_seg_gan/PITI/logs/coco-64-stage1/
export PYTHONPATH=$PYTHONPATH:$(pwd)
NUM_GPUS=8
MODEL_FLAGS="--learn_sigma True --uncond_p 0. --image_size 64 --finetune_decoder False"
TRAIN_FLAGS="--lr 3.5e-5 --batch_size 16  --schedule_sampler loss-second-moment  --model_path /home/d-d-sh/adv_seg_gan/PITI/ckpt/base_mask.pt --lr_anneal_steps 200000"
DIFFUSION_FLAGS=""
SAMPLE_FLAGS="--num_samples 2 --sample_c 1"
DATASET_FLAGS="--data_dir /home/d-d-sh/adv_seg_gan/PITI/dataset/coco_train_ann_color.txt --val_data_dir /home/d-d-sh/adv_seg_gan/PITI/dataset/coco_val_ann_color.txt --mode coco"
mpiexec -n $NUM_GPUS --allow-run-as-root python ./image_train.py $MODEL_FLAGS  $TRAIN_FLAGS $SAMPLE_FLAGS $DIFFUSION_FLAGS  $DATASET_FLAGS
# python ./image_train.py $MODEL_FLAGS  $TRAIN_FLAGS $SAMPLE_FLAGS $DIFFUSION_FLAGS  $DATASET_FLAGS


# export CUDA_LAUNCH_BLOCKING=1
# export LOGDIR=/home/d-d-sh/adv_seg_gan/PITI/logs/coco-64-stage1-cont/
# export PYTHONPATH=$PYTHONPATH:$(pwd)
# NUM_GPUS=8
# MODEL_FLAGS="--learn_sigma True --uncond_p 0.2 --image_size 64 --finetune_decoder False --encoder_path /home/d-d-sh/adv_seg_gan/PITI/logs/coco-64-stage1/checkpoints/ema_0.9999_200000.pt"
# TRAIN_FLAGS="--lr 2e-5 --batch_size 16  --schedule_sampler loss-second-moment  --model_path /home/d-d-sh/adv_seg_gan/PITI/ckpt/base_mask.pt --lr_anneal_steps 60000"
# DIFFUSION_FLAGS=""
# SAMPLE_FLAGS="--num_samples 2 --sample_c 1.5"
# DATASET_FLAGS="--data_dir /home/d-d-sh/adv_seg_gan/PITI/dataset/coco_train_ann_color.txt --val_data_dir /home/d-d-sh/adv_seg_gan/PITI/dataset/coco_val_ann_color.txt --mode coco"
# mpiexec -n $NUM_GPUS --allow-run-as-root python ./image_train.py $MODEL_FLAGS  $TRAIN_FLAGS $SAMPLE_FLAGS $DIFFUSION_FLAGS  $DATASET_FLAGS
# # python ./image_train.py $MODEL_FLAGS  $TRAIN_FLAGS $SAMPLE_FLAGS $DIFFUSION_FLAGS  $DATASET_FLAGS


# export LOGDIR=/home/d-d-sh/adv_seg_gan/PITI/logs/coco-64-stage2-decoder/
# export PYTHONPATH=$PYTHONPATH:$(pwd)
# NUM_GPUS=8
# MODEL_FLAGS="--learn_sigma True --uncond_p 0.2 --image_size 64 --finetune_decoder True"
# TRAIN_FLAGS="--lr 3.5e-5 --batch_size 16 --schedule_sampler loss-second-moment --model_path /home/d-d-sh/adv_seg_gan/PITI/ckpt/base_mask.pt --encoder_path /home/d-d-sh/adv_seg_gan/PITI/logs/coco-64-stage1-cont/checkpoints/ema_0.9999_060000.pt"
# DIFFUSION_FLAGS=""
# SAMPLE_FLAGS="--num_samples 2 --sample_c 1.5"
# DATASET_FLAGS="--data_dir /home/d-d-sh/adv_seg_gan/PITI/dataset/coco_train_ann_color.txt --val_data_dir /home/d-d-sh/adv_seg_gan/PITI/dataset/coco_val_ann_color.txt --mode coco"
# mpiexec -n $NUM_GPUS --allow-run-as-root python ./image_train.py $MODEL_FLAGS  $TRAIN_FLAGS $SAMPLE_FLAGS $DIFFUSION_FLAGS  $DATASET_FLAGS
# # python ./image_train.py $MODEL_FLAGS  $TRAIN_FLAGS $SAMPLE_FLAGS $DIFFUSION_FLAGS  $DATASET_FLAGS
 
 