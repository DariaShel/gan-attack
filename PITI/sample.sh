# # sketch to image
# export LOGDIR=./test/sketch/
# export PYTHONPATH=$PYTHONPATH:$(pwd)
# NUM_GPUS=1
# MODEL_FLAGS="--learn_sigma True --model_path ./ckpt/base_edge.pt --sr_model_path ./ckpt/upsample_edge.pt"
# SAMPLE_FLAGS="--num_samples 5000 --sample_c 1.3 --batch_size 50"
# DATASET_FLAGS="--data_dir ./dataset/COCOSTUFF_train.txt --val_data_dir ./dataset/COCOSTUFF_val.txt --mode coco-edge"
# mpiexec -n $NUM_GPUS --allow-run-as-root python ./image_sample.py $MODEL_FLAGS $SAMPLE_FLAGS  $DATASET_FLAGS
 

# mask to image
export CUDA_VISIBLE_DEVICES=2
export LOGDIR=./test/mask/
export PYTHONPATH=$PYTHONPATH:$(pwd)
NUM_GPUS=1
MODEL_FLAGS="--learn_sigma True --model_path ./ckpt/base_mask.pt --sr_model_path ./ckpt/upsample_mask.pt"
SAMPLE_FLAGS="--num_samples 5000 --sample_c 1.3 --batch_size 1"
DATASET_FLAGS="--data_dir /home/d-d-sh/adv_seg_gan/PITI/dataset/coco_val100.txt --val_data_dir /home/d-d-sh/adv_seg_gan/PITI/dataset/coco_val100.txt --mode coco"
nohup mpiexec -n $NUM_GPUS --allow-run-as-root python ./image_sample.py $MODEL_FLAGS $SAMPLE_FLAGS  $DATASET_FLAGS > nohup_fid100.out 2>&1 &
 
