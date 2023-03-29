
export LOGDIR=/home/d-d-sh/adv_seg_gan/PITI/logs/coco-upsample/ 
export PYTHONPATH=$PYTHONPATH:$(pwd)
NUM_GPUS=3
MODEL_FLAGS="--learn_sigma True --uncond_p 0 --image_size 256 --super_res 64 --num_res_blocks 2 --finetune_decoder True --model_path /home/d-d-sh/adv_seg_gan/PITI/ckpt/upsample_mask.pt --lr_anneal_steps 2000"
TRAIN_FLAGS="--lr 1e-5 --batch_size 1"
DIFFUSION_FLAGS="--noise_schedule linear"
SAMPLE_FLAGS="--num_samples 2 --sample_c 1"
DATASET_FLAGS="--data_dir /home/d-d-sh/adv_seg_gan/PITI/dataset/coco_val200_1.txt --val_data_dir /home/d-d-sh/adv_seg_gan/PITI/dataset/coco_val200_1.txt --mode coco"
nohup mpiexec -n $NUM_GPUS --allow-run-as-root python ./attack.py $MODEL_FLAGS  $TRAIN_FLAGS $SAMPLE_FLAGS $DIFFUSION_FLAGS $DATASET_FLAGS &!


 