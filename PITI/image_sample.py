"""
Train a diffusion model on images.
"""
import argparse

from pretrained_diffusion import dist_util, logger
from pretrained_diffusion.image_datasets_mask import load_data_mask
from pretrained_diffusion.image_datasets_sketch import load_data_sketch
from pretrained_diffusion.image_datasets_depth import load_data_depth
from pretrained_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from pretrained_diffusion.train_util import TrainLoop
from pretrained_diffusion.glide_util import sample 
import torch
import os
import torch as th
import torchvision.utils as tvu
import torch.distributed as dist
import numpy as np
import pytorch_fid_wrapper as pfw
import pandas as pd
from tqdm import tqdm

def get_mask(gt_mask):
        classes = np.unique(gt_mask)
        masks = []
        for i, cl in enumerate(classes):
            mask = np.where(gt_mask == cl, 1, 0)
            mask = np.expand_dims(mask, axis=(0, 3))
            masks.append(mask)
        masks = np.vstack(masks)
        areas = np.sum(masks, axis=(1, 2, 3))
        idx = np.flip(np.argsort(areas))[0]

        gt_mask_tensor = th.from_numpy(gt_mask)
        mask_overlay = th.where(gt_mask_tensor == classes[idx], 1, 0)

        return mask_overlay.int().unsqueeze(0)

def main():
    parser, parser_up = create_argparser()
    
    args = parser.parse_args()
    args_up = parser_up.parse_args()
    dist_util.setup_dist()

    options=args_to_dict(args, model_and_diffusion_defaults(0.).keys())
    model, diffusion = create_model_and_diffusion(**options)
 
    options_up=args_to_dict(args_up, model_and_diffusion_defaults(True).keys())
    model_up, diffusion_up = create_model_and_diffusion(**options_up)
 

    if  args.model_path:
        print('loading model')
        model_ckpt = dist_util.load_state_dict(args.model_path, map_location="cpu")

        model.load_state_dict(
            model_ckpt   , strict=True )

    if  args.sr_model_path:
        print('loading sr model')
        model_ckpt2 = dist_util.load_state_dict(args.sr_model_path, map_location="cpu")

        model_up.load_state_dict(
            model_ckpt2   , strict=True ) 

 
    model.to(dist_util.dev())
    model_up.to(dist_util.dev())
    model.eval()
    model_up.eval()
 
########### dataset
    logger.log("creating data loader...")

    if args.mode == 'ade20k' or args.mode == 'coco':
 
        val_data = load_data_mask(
            data_dir=args.val_data_dir,
            batch_size=args.batch_size,
            image_size=256,
            train=False,
            deterministic=True,
            low_res=args.super_res,
            uncond_p = 0. ,
            mode = args.mode,
            random_crop=False,
        )

    elif args.mode == 'depth' or args.mode == 'depth-normal':
        val_data = load_data_depth(
            data_dir=args.val_data_dir,
            batch_size=args.batch_size//2,
            image_size=256,
            train=False,
            deterministic=True,
            low_res=args.super_res,
            uncond_p = 0. ,
            mode = args.mode,
            random_crop=False,
        )    


    elif args.mode == 'coco-edge' or args.mode == 'flickr-edge':
        val_data = load_data_sketch(
            data_dir=args.val_data_dir,
            batch_size=args.batch_size//2,
            image_size=256,
            train=False,
            deterministic=True,
            low_res=args.super_res,
            uncond_p = 0. ,
            mode = args.mode,
            random_crop=False,
        )    


 
    logger.log("sampling...")
    gt_path = os.path.join(logger.get_dir(), 'GT')
    os.makedirs(gt_path,exist_ok=True)
    lr_path = os.path.join(logger.get_dir(), 'LR')
    os.makedirs(lr_path,exist_ok=True)    
    hr_path = os.path.join(logger.get_dir(), 'HR')
    os.makedirs(hr_path,exist_ok=True)
    ref_path = os.path.join(logger.get_dir(), 'REF')
    os.makedirs(ref_path,exist_ok=True)    

    img_id = 0
    n = 100
    all_overlay_images = []
    all_generated_images = []
    all_clean_images = []
    for i in tqdm(range(n), total=n, desc='FID'):

        batch, model_kwargs = next(val_data)  
        mask_overlay = get_mask(model_kwargs['mask'][0].numpy())  
          
        with th.no_grad():
            samples_lr =sample(
                glide_model= model,
                glide_options= options,
                side_x= 64,
                side_y= 64,
                prompt=model_kwargs,
                batch_size= args.batch_size,
                guidance_scale=args.sample_c,
                device=dist_util.dev(),
                prediction_respacing= "250",
                upsample_enabled= False,
                upsample_temp=0.997,
                mode = args.mode,
            )

            samples_lr = samples_lr.clamp(-1, 1)

            tmp = (127.5*(samples_lr + 1.0)).int() 
            model_kwargs['low_res'] = tmp/127.5 - 1.

            samples_hr =sample(
                glide_model= model_up,
                glide_options= options_up,
                side_x=256,
                side_y=256,
                prompt=model_kwargs,
                batch_size=args.batch_size,
                guidance_scale=1,
                device=dist_util.dev(),
                prediction_respacing= "fast27",
                upsample_enabled=True,
                upsample_temp=0.997,
                mode = args.mode,
            )


            samples_lr = samples_lr.cpu()
            # ref = model_kwargs['ref'].cpu()
            ref =  model_kwargs['ref_ori'].cpu()
       
            samples_hr = samples_hr.cpu()

            overlay_img = th.where(mask_overlay.to(dist_util.dev()) == 1, samples_hr[0].to(dist_util.dev()), batch[0].to(dist_util.dev()))
            if overlay_img.shape[0] == 1:
                overlay_img = torch.tile(overlay_img, (3, 1, 1))

            if samples_hr[0].shape[0] == 1:
                generated = torch.tile(generated, (3, 1, 1))
            else:
                generated = samples_hr[0]
                
            if batch[0].shape[0] == 1:
                orig = torch.tile(batch[0], (3, 1, 1))
            else:
                orig = batch[0]

            all_overlay_images.append(overlay_img.unsqueeze(0).cpu().detach())
            all_generated_images.append(generated.unsqueeze(0).cpu().detach())
            all_clean_images.append(orig.unsqueeze(0).cpu().detach())

            # for i in range(samples_lr.size(0)):
            #     name = model_kwargs['path'][i].split('/')[-1].split('.')[0] + '.png'
            #     out_path = os.path.join(lr_path, name)
            #     tvu.save_image(
            #         (samples_lr[i]+1)*0.5, out_path)

            #     out_path = os.path.join(gt_path, name)
            #     tvu.save_image(
            #         (batch[i]+1)*0.5, out_path)

            #     out_path = os.path.join(ref_path, name)
            #     tvu.save_image(
            #         (ref[i]+1)*0.5, out_path)

      
            #     out_path = os.path.join(hr_path, name)
            #     tvu.save_image(
            #         (samples_hr[i]+1)*0.5, out_path)

            #     img_id += 1  
    all_overlay_images = torch.nan_to_num(torch.cat(all_overlay_images))
    all_generated_images = torch.nan_to_num(torch.cat(all_generated_images))
    all_clean_images = torch.nan_to_num(torch.cat(all_clean_images))

    pfw.set_config(batch_size=20, device=dist_util.dev())
    try:
        fid_full_genegated = pfw.fid(all_generated_images, all_clean_images)
    except ValueError:
        fid_full_genegated = -1
    try:
        fid_segments_generated = pfw.fid(all_overlay_images, all_clean_images)
    except ValueError:
        fid_segments_generated = -1

    print()
    print()
    print()
    print(f"FID_full_generated: {fid_full_genegated}")
    print(f"FID_segments_generated: {fid_segments_generated}")

    results = pd.DataFrame(
        {
            "FID_full_generated": [fid_full_genegated],
            "FID_segments_generated": [fid_segments_generated],
        }
    )

    results.to_csv(f'FID_no_attack100.csv')
    


def create_argparser():
    defaults = dict(
        data_dir="",
        val_data_dir="",
        model_path="",
        sr_model_path="",
        encoder_path="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=20000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        sample_c=1.,
        sample_respacing="100",
        uncond_p=0.2,
        num_samples=1,
        finetune_decoder = False,
        mode = '',
        )

    defaults_up = defaults
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    defaults_up.update(model_and_diffusion_defaults(True))
    parser_up = argparse.ArgumentParser()
    add_dict_to_argparser(parser_up, defaults_up)

    return parser, parser_up


if __name__ == "__main__":
    main()
 
 