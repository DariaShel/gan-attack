"""
Train a diffusion model on images.
"""
import argparse
import torch.distributed as dist
from pretrained_diffusion import dist_util, logger
from pretrained_diffusion.image_datasets_mask import load_data_mask
from pretrained_diffusion.image_datasets_sketch import load_data_sketch
from pretrained_diffusion.image_datasets_depth import load_data_depth
from pretrained_diffusion.resample import create_named_schedule_sampler
from pretrained_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,)
from pretrained_diffusion.attack_util import TrainLoop
import torch

from mmdet.apis import init_detector
from tqdm import tqdm

import numpy as np
from mmdet.core.evaluation.mean_ap import eval_map
from torchmetrics.image.fid import FrechetInceptionDistance
from inception_score import inception_score
import pytorch_fid_wrapper as pfw
import pandas as pd
import json
from pathlib import Path
from mpi4py import MPI

def get_gt_annotation(gt_norm, path):
    img_name = Path(path).name

    bboxes = []
    labels = []

    if gt_norm[img_name].keys():
        for label, objects in gt_norm[img_name].items():
            for obj in objects:
                labels.append(label)
                bboxes.append(obj)
        gt_labels = np.array(labels)
        gt_bboxes = np.vstack(bboxes)

        gt_info = {
            "bboxes": gt_bboxes,
            "labels": gt_labels,
        }
        return [gt_info]

    else:
        return None


def get_bboxes(model_output):
    det_bboxes = []
    for cls in model_output:
        if cls.shape[0] == 0:
            det_bboxes.append(cls)
        else:
            cls_bboxes = cls[:, :-1]
            det_bboxes.append(cls_bboxes)

    return det_bboxes

def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0      
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:        
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)


def denormalize(image_tensor):
    if isinstance(image_tensor, list):
        denorm_image = []
        for i in range(len(image_tensor)):
            denorm_image.append(denormalize(image_tensor[i]))
        return denorm_image
    denorm_image = image_tensor.clone()
    
    denorm_image = (denorm_image + 1) / 2.0 * 255.0  
    denorm_image = torch.clip(denorm_image, 0, 255)

    return denorm_image

class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()

    options=args_to_dict(args, model_and_diffusion_defaults(args.super_res).keys())
    model, diffusion = create_model_and_diffusion(**options)

    options=args_to_dict(args)
    if dist.get_rank() == 0:
        logger.save_args(options)
    
    config_file = '/home/d-d-sh/adv_seg_gan/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint_file = '/home/d-d-sh/adv_seg_gan/PITI/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

    # build the model from a config file and a checkpoint file
    detector = init_detector(config_file, checkpoint_file, device=dist_util.dev())


########### dataset selection
    logger.log("creating data loader...")
    if args.mode == 'ade20k' or args.mode == 'coco':
        data = load_data_mask(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            train=True,
            low_res=args.super_res,
            uncond_p = args.uncond_p,
            mode = args.mode,
            random_crop=False,
        )

        val_data = load_data_mask(
            data_dir=args.val_data_dir,
            batch_size=args.batch_size//2,
            image_size=args.image_size,
            train=False,
            deterministic=True,
            low_res=args.super_res,
            uncond_p = 0. ,
            mode = args.mode,
            random_crop=False,
        )

    elif args.mode == 'depth' or args.mode == 'depth-normal':
        data = load_data_depth(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            train=True,
            low_res=args.super_res,
            uncond_p = args.uncond_p,
            mode = args.mode,
            random_crop=True,
        )

        val_data = load_data_depth(
            data_dir=args.val_data_dir,
            batch_size=args.batch_size//2,
            image_size=args.image_size,
            train=False,
            deterministic=True,
            low_res=args.super_res,
            uncond_p = 0. ,
            mode = args.mode,
            random_crop=False,
        )    


    elif args.mode == 'coco-edge' or args.mode == 'flickr-edge':
        data = load_data_sketch(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            train=True,
            low_res=args.super_res,
            uncond_p = args.uncond_p,
            mode = args.mode,
            random_crop=True,
        )

        val_data = load_data_sketch(
            data_dir=args.val_data_dir,
            batch_size=args.batch_size//2,
            image_size=args.image_size,
            train=False,
            deterministic=True,
            low_res=args.super_res,
            uncond_p = 0. ,
            mode = args.mode,
            random_crop=False,
        )
    
    image, model_kwargs = next(data, (None, None))
    n = 200
    mean_ap_clean_total = AverageMeter()
    mean_ap_attacked_total = AverageMeter()
    attack_images = []
    generated_images = []
    orig_images = []

    all_attack_images = []
    all_generated_images = []
    all_clean_images = []

    with open("gt_coco_for_mmdet.json") as json_file:
        gt_norm = json.load(json_file, object_hook=lambda d: {int(k) if k.lstrip('-').isdigit() else k: v for k, v in d.items()})

    for i in tqdm(range(n), total=n, desc='Attack'):
        ##### scratch #####
        if args.model_path:
            print('loading decoder')
            model_ckpt = dist_util.load_state_dict(args.model_path, map_location="cpu")

            for k  in list(model_ckpt.keys()):
                if k.startswith("transformer") and 'transformer_proj'  not in k:
                    # print(f"Removing key {k} from pretrained checkpoint")
                    del model_ckpt[k]
                if k.startswith("padding_embedding") or k.startswith("positional_embedding") or k.startswith("token_embedding") or k.startswith("final_ln"):
                    # print(f"Removing key {k} from pretrained checkpoint")
                    del model_ckpt[k]

            model.load_state_dict(
                model_ckpt, strict=True )


        if args.encoder_path:
            print('loading encoder')
            encoder_ckpt = dist_util.load_state_dict(args.encoder_path, map_location="cpu")
            model.encoder.load_state_dict(
                encoder_ckpt   , strict=True )        

        model.to(dist_util.dev())
        schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion) 
        logger.log("training...")

        pred_det_clean, pred_det_attack, generated, overlay_img, clean_img, path = TrainLoop(
            model,
            options,
            model_kwargs,
            diffusion,
            detector,
            data=image,
            val_data=image,
            img_id=i,
            batch_size=args.batch_size,
            microbatch=args.microbatch,
            lr=args.lr,
            ema_rate=args.ema_rate,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            resume_checkpoint=args.resume_checkpoint,
            use_fp16=args.use_fp16,
            fp16_scale_growth=args.fp16_scale_growth,
            schedule_sampler=schedule_sampler,
            weight_decay=args.weight_decay,
            lr_anneal_steps=args.lr_anneal_steps,
            finetune_decoder = args.finetune_decoder,
            mode =  args.mode,
            use_vgg = args.super_res,
            use_gan = args.super_res,
            uncond_p = args.uncond_p,
            super_res = args.super_res,
        ).run_loop()

        if overlay_img.shape[0] == 1:
            overlay_img = torch.tile(overlay_img, (3, 1, 1))

        if generated.shape[0] == 1:
            generated = torch.tile(generated, (3, 1, 1))
            
        if clean_img.shape[0] == 1:
            orig = torch.tile(clean_img, (3, 1, 1))
        else:
            orig = clean_img.clone()

        all_attack_images.append(overlay_img.unsqueeze(0).cpu().detach())
        all_generated_images.append(generated.unsqueeze(0).cpu().detach())
        all_clean_images.append(orig.unsqueeze(0).cpu().detach())

        attack_images.append(denormalize(overlay_img).unsqueeze(0).type(torch.uint8).cpu().detach())
        generated_images.append(denormalize(generated).unsqueeze(0).type(torch.uint8).cpu().detach())
        orig_images.append(denormalize(orig).unsqueeze(0).type(torch.uint8).cpu().detach())

        annotation = get_gt_annotation(gt_norm, path)
        if annotation is not None:
            det_bboxes_orig = get_bboxes(pred_det_clean)
            det_bboxes_attacked = get_bboxes(pred_det_attack)

            mean_ap_clean, _ = eval_map(
                det_results=[det_bboxes_orig], annotations=annotation
            )
            mean_ap_attacked, _ = eval_map(
                det_results=[det_bboxes_attacked], annotations=annotation
            )

            mean_ap_clean_total.update(mean_ap_clean)
            mean_ap_attacked_total.update(mean_ap_attacked)

        image, model_kwargs = next(data, (None, None))
        print(f'{i + 1}/{n}\n')

    all_attack_images = torch.nan_to_num(torch.cat(all_attack_images))
    all_generated_images = torch.nan_to_num(torch.cat(all_generated_images))
    all_clean_images = torch.nan_to_num(torch.cat(all_clean_images))

    pfw.set_config(batch_size=20, device=dist_util.dev())
    try:
        fid_full_genegated = pfw.fid(all_generated_images, all_clean_images)
    except ValueError:
        fid_full_genegated = -1
    try:
        fid_segments_generated = pfw.fid(all_attack_images, all_clean_images)
    except ValueError:
        fid_segments_generated = -1

    is_full_generated = inception_score(all_generated_images, resize=True, batch_size=20, cuda=True)[0]
    is_segments_generated = inception_score(all_attack_images, resize=True, batch_size=20, cuda=True)[0]

    print()
    print()
    print()
    print(f"Original_images_mAP: {mean_ap_clean_total.avg}")
    print(f'Attacked_images_mAP: {mean_ap_attacked_total.avg}')
    print(f"FID_full_generated: {fid_full_genegated}")
    print(f"FID_segments_generated: {fid_segments_generated}")
    print(f"IS_full_generated: {is_full_generated.item()}")
    print(f"IS_segments_generated: {is_segments_generated.item()}")

    results = pd.DataFrame(
        {
            "Original_images_mAP": [mean_ap_clean_total.avg],
            "Attacked_images_mAP": [mean_ap_attacked_total.avg],
            "FID_full_generated": [fid_full_genegated],
            "FID_segments_generated": [fid_segments_generated],
            "IS_full_generated": [is_full_generated.item()],
            "IS_segments_generated": [is_segments_generated.item()],
        }
    )

    results.to_csv(f'proc_{MPI.COMM_WORLD.Get_rank()}_success_attack_diffusion200_1.csv')

 
def create_argparser():
    defaults = dict(
        data_dir="",
        val_data_dir="",
        model_path="",
        encoder_path="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=400,
        save_interval=200,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        super_res=0,
        sample_c=1.,
        sample_respacing="100",
        uncond_p=0.2,
        num_samples=1,
        finetune_decoder = False,
        mode =  "",
        )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
