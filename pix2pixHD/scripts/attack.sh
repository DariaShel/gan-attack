nohup python attack_gan.py --name attack1 --netG local --ngf 32 --num_D 3 --load_pretrain checkpoints/orig/ --niter 30 --niter_decay 10 --niter_fix_global 20 --resize_or_crop none --no_ganFeat_loss --lr 0.0005 --no_flip > nohup_val.out 2>&1 &