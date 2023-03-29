import time
import os
from tqdm import tqdm
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
import fractions

import torchvision.transforms as transforms


def lcm(a, b):
    return abs(a * b) / fractions.gcd(a, b) if a and b else 0


from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from inception_score import inception_score

import cv2
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from attack_utils import (
    AverageMeter,
    get_bboxes,
    get_detector_loss,
    get_gt_annotation,
    get_max_segment,
    det_output2numpy,
    overlay,
    get_bboxes_with_probs,
)
from mmdet.core.evaluation.mean_ap import eval_map
from torchmetrics.image.fid import FrechetInceptionDistance
import pandas as pd
from pathlib import Path
import pytorch_fid_wrapper as pfw


# import argparse
# import multiprocessing as mp
# # fmt: off
# import sys
# sys.path.insert(1, os.path.join(sys.path[0], '..'))
# # import sys
# # fmt: on

# from detectron2.config import get_cfg
# from detectron2.data.detection_utils import read_image
# from detectron2.projects.deeplab import add_deeplab_config
# from detectron2.utils.logger import setup_logger

# sys.path.append(r'/home/d-d-sh/adv_seg_gan/pix2pixHD/OneFormer')
# from oneformer import (
#     add_oneformer_config,
#     add_common_config,
#     add_swin_config,
#     add_dinat_config,
#     add_convnext_config,
# )
# sys.path.append(r'/home/d-d-sh/adv_seg_gan/pix2pixHD/OneFormer/demo')
# from predictor import VisualizationDemo

# # constants
# WINDOW_NAME = "OneFormer Demo"

# def setup_cfg(args):
#     # load config from file and command-line arguments
#     cfg = get_cfg()
#     add_deeplab_config(cfg)
#     add_common_config(cfg)
#     add_swin_config(cfg)
#     add_dinat_config(cfg)
#     add_convnext_config(cfg)
#     add_oneformer_config(cfg)
#     cfg.merge_from_file(args.config_file)
#     cfg.merge_from_list(args.opts)
#     cfg.freeze()
#     return cfg


# seg_args = Bunch()
# seg_args.config_file = "/home/d-d-sh/adv_seg_gan/pix2pixHD/OneFormer/configs/cityscapes/convnext/oneformer_convnext_xlarge_bs16_90k.yaml"
# seg_args.confidence_threxhold = 0.5
# seg_args.opts = ['MODEL.IS_TRAIN', 'False', 'MODEL.IS_DEMO', 'True', 'MODEL.WEIGHTS', '/home/d-d-sh/adv_seg_gan/pix2pixHD/OneFormer/250_16_convnext_xl_oneformer_cityscapes_90k.pth']


# def get_parser():
#     parser = argparse.ArgumentParser(description="oneformer demo for builtin configs")
#     parser.add_argument(
#         "--config-file",
#         default="/home/d-d-sh/adv_seg_gan/pix2pixHD/OneFormer/configs/cityscapes/convnext/oneformer_convnext_xlarge_bs16_90k.yaml",
#         metavar="FILE",
#         help="path to config file",
#     )
#     parser.add_argument("--task", help="Task type")
#     parser.add_argument(
#         "--input",
#         default='',
#         nargs="+",
#         help="A list of space separated input images; "
#         "or a single glob pattern such as 'directory/*.jpg'",
#     )
#     parser.add_argument(
#         "--output",
#         default='',
#         help="A file or directory to save output visualizations. "
#         "If not given, will show output in an OpenCV window.",
#     )

#     parser.add_argument(
#         "--confidence-threshold",
#         type=float,
#         default=0.5,
#         help="Minimum score for instance predictions to be shown",
#     )
#     parser.add_argument(
#         "--opts",
#         help="Modify config options using the command-line 'KEY VALUE' pairs",
#         default=[],
#         nargs=argparse.REMAINDER,
#     )
#     return parser

config_file_det = "/home/d-d-sh/adv_seg_gan/mmdetection/configs/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes.py"
checkpoint_file_det = (
    "/home/d-d-sh/adv_seg_gan/faster_rcnn_r50_fpn_1x_cityscapes_20200502-829424c0.pth"
)
detector = init_detector(config_file_det, checkpoint_file_det, device="cuda:0")

opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, "iter.txt")

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print("#training images = %d" % dataset_size)

visualizer = Visualizer(opt)

attack_segments_id = {"road": 7, "building": 11, "sky": 23}

detector_classes = {
    "person": 0,
    "rider": 1,
    "car": 2,
    "truck": 3,
    "bus": 4,
    "train": 5,
    "motorcycle": 6,
    "bicycle": 7,
}

criterion = torch.nn.BCELoss()

# annotations_dir = "/home/d-d-sh/datasets/cityscapes/data_for_evaluate_detector/annotations"
annotations_dir = "/home/d-d-sh/datasets/cityscapes/data_for_evaluate_detector_val/annotations"

mean_ap_clean_total = AverageMeter()
mean_ap_attacked_total = AverageMeter()
is_full_generated_total = AverageMeter()
is_segments_generated_total = AverageMeter()
# fid_full_generated_total = AverageMeter()
# fid_segments_generated_total = AverageMeter()
attack_images = []
generated_images = []
orig_images = []

all_attack_images = []
all_generated_images = []
all_clean_images = []

# minibatch = 0

for i, data in tqdm(enumerate(dataset), desc="Batch of images", total=len(dataset)):
    # mp.set_start_method("spawn", force=True)
    # # args = get_parser().parse_args()
    # setup_logger(name="fvcore")
    # logger = setup_logger()
    # logger.info("Arguments: " + str(seg_args))
    # seg_args.input = data['path'][0]
    # cfg = setup_cfg(seg_args)
    # demo = VisualizationDemo(cfg)

    # img = read_image(seg_args.input, format="BGR")
    # predictions, visualized_output, mask_semantic = demo.run_on_image(img, 'semantic')
    # predictions, visualized_output, mask_instance = demo.run_on_image(img, 'instance', mask_semantic)

    # cv2.imwrite(f'results/detecting_results2/mask_semantic{i}.png', mask_semantic.detach().cpu().numpy())
    # cv2.imwrite(f'results/detecting_results2/mask_instance{i}.png', mask_instance.detach().cpu().numpy())

    # mask_semantic = mask_semantic.reshape(1, 1, mask_semantic.shape[0], mask_semantic.shape[1])
    # mask_instance = mask_instance.reshape(1, 1, mask_instance.shape[0], mask_instance.shape[1])

    # print(torch.unique(mask_semantic))
    # print(torch.unique(mask_instance))
    # print(torch.unique(data['inst']))


    model = create_model(opt)
    if opt.fp16:
        from apex import amp

        model, [optimizer_G, optimizer_D] = amp.initialize(
            model, [model.optimizer_G, model.optimizer_D], opt_level="O1"
        )
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
    else:
        optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D

    det_result_orig = inference_detector(
        detector, util.denormalize(data["image"][0]).cuda()
    )
    _, probs = get_bboxes_with_probs(
        det_result_orig, [detector_classes["car"], detector_classes["person"]]
    )
    if len(probs) == 0:
        continue
    det_result_orig_numpy = det_output2numpy(det_result_orig)
    detecting_clean_image = show_result_pyplot(
        detector,
        util.tensor2im(data["image"][0].data),
        det_result_orig_numpy,
        show=False,
    )

    for iter in tqdm(
        range(opt.niter + opt.niter_decay + 1),
        desc="Attack",
        total=opt.niter + opt.niter_decay + 1,
    ):
        iter_start_time = time.time()

        ############# Forward Pass ######################
        losses, generated = model(
            Variable(data["label"]),
            Variable(data["inst"]),
            Variable(data["image"]),
            Variable(data["feat"]),
            infer=True,
        )



        # losses, generated = model(
        #     Variable(mask_semantic),
        #     Variable(mask_instance),
        #     Variable(data["image"]),
        #     Variable(data["feat"]),
        #     infer=True,
        # )

        # cv2.imwrite(f'results/detecting_results2/generated{iter}.png', util.tensor2im(generated[0].detach().cpu())[:, :, ::-1])
        # cv2.imwrite(f'results/detecting_results2/original12345678.png', util.tensor2im(data["image"][0].detach().cpu())[:, :, ::-1])
        # print(data["path"][0])
        # break
        mask_overlay = get_max_segment(data["label"][0], attack_segments_id)
        # mask_overlay = get_max_segment(mask_semantic[0], attack_segments_id)

        attack_img = overlay(
            data["image"][0], generated[0], mask_overlay
        )
        det_result = inference_detector(detector, util.denormalize(attack_img))
        det_result_numpy = det_output2numpy(det_result)
        attack_img_np = util.tensor2im(attack_img.data)
        detect_attacked_image = show_result_pyplot(
            detector, attack_img_np, det_result_numpy, show=False
        )

        _, probs = get_bboxes_with_probs(
            det_result, [detector_classes["car"], detector_classes["person"]]
        )
        if len(probs) == 0:
            break

        detector_loss_G = get_detector_loss(probs)

        encode_label = model.module.encode_input(data["label"], data["inst"])[0]
        # encode_label = model.module.encode_input(mask_semantic, mask_instance)[0]
        pred_discriminator = model.module.discriminate(
            encode_label, attack_img.unsqueeze(0), use_pool=True
        )
        detector_loss_D = model.module.criterionGAN(pred_discriminator, True)

        # sum per device losses
        losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
        loss_dict = dict(zip(model.module.loss_names, losses))

        # calculate final loss scalar
        loss_D = (loss_dict["D_fake"] + loss_dict["D_real"]) * 0.5
        loss_G = (
            loss_dict["G_GAN"]
            + loss_dict.get("G_VGG", 0)
            + detector_loss_G
            + detector_loss_D
        )

        ############### Backward Pass ####################
        # update generator weights
        optimizer_G.zero_grad()
        if opt.fp16:
            with amp.scale_loss(loss_G, optimizer_G) as scaled_loss:
                scaled_loss.backward()
        else:
            loss_G.backward()
        optimizer_G.step()

        # update discriminator weights
        optimizer_D.zero_grad()
        if opt.fp16:
            with amp.scale_loss(loss_D, optimizer_D) as scaled_loss:
                scaled_loss.backward()
        else:
            loss_D.backward()
        optimizer_D.step()

        ############## Display results and errors ##########
        ### print out errors
        errors = {
            k: v.data.item() if not isinstance(v, int) else v
            for k, v in loss_dict.items()
        }
        errors["detector_loss_D"] = detector_loss_D.item()
        errors["detector_loss_G"] = detector_loss_G.item()
        t = time.time() - iter_start_time
        visualizer.print_current_errors(i, iter, errors, t)
        visualizer.plot_current_errors(errors, step=iter)

        # call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])

        ### display output images
        visuals = OrderedDict(
            [
                ("input_label", util.tensor2label(data["label"][0], opt.label_nc)),
                # ("input_label", util.tensor2label(mask_semantic[0], opt.label_nc)),
                ("synthesized_image", util.tensor2im(generated.data[0])),
                ("real_image", util.tensor2im(data["image"][0])),
                ("attack_image", attack_img_np),
                ("detecting_attack_image", detect_attacked_image),
                ("detecting_clean_image", detecting_clean_image),
            ]
        )

        visualizer.display_current_results(visuals, iter, iter)

        ### instead of only training the local enhancer, train the entire network after certain iterations
        if (opt.niter_fix_global != 0) and (iter == opt.niter_fix_global):
            model.module.update_fixed_params()

        ### linearly decay learning rate after certain iterations
        if iter > opt.niter:
            model.module.update_learning_rate()


    if attack_img.shape[0] == 1:
        attack_img = torch.tile(attack_img, (3, 1, 1))

    if generated.shape[1] == 1:
        generated = torch.tile(generated, (1, 3, 1, 1))
        
    if data["image"].shape[1] == 1:
        orig = torch.tile(data["image"], (1, 3, 1, 1))
    else:
        orig = data["image"].clone()

    all_attack_images.append(attack_img.unsqueeze(0).cpu().detach())
    all_generated_images.append(generated.cpu().detach())
    all_clean_images.append(orig.cpu().detach())

    attack_images.append(util.denormalize(attack_img).unsqueeze(0).type(torch.uint8).cpu().detach())
    generated_images.append(util.denormalize(generated).type(torch.uint8).cpu().detach())
    orig_images.append(util.denormalize(orig).type(torch.uint8).cpu().detach())

    # minibatch += 1
    # print(data["path"])
    annotation = get_gt_annotation(annotations_dir, data["path"][0], detector_classes)

    if annotation is not None:
        det_bboxes_orig = get_bboxes(det_result_orig_numpy)
        det_bboxes_attacked = get_bboxes(det_result_numpy)

        mean_ap_clean, _ = eval_map(
            det_results=[det_bboxes_orig], annotations=annotation
        )
        mean_ap_attacked, _ = eval_map(
            det_results=[det_bboxes_attacked], annotations=annotation
        )

        mean_ap_clean_total.update(mean_ap_clean)
        mean_ap_attacked_total.update(mean_ap_attacked)

    # cv2.imwrite(f'results/detecting_results3/original_image{i}.png', detecting_clean_image[:, :, ::-1])
    # cv2.imwrite(f'results/detecting_results3/attacked_image{i}.png', detect_attacked_image[:, :, ::-1])
    # cv2.imwrite(f'results/detecting_results2/generated{i}.png', util.tensor2im(generated[0].detach().cpu())[:, :, ::-1])

all_attack_images = torch.nan_to_num(torch.cat(all_attack_images))
all_generated_images = torch.nan_to_num(torch.cat(all_generated_images))
all_clean_images = torch.nan_to_num(torch.cat(all_clean_images))

pfw.set_config(batch_size=1)
try:
    fid_full_genegated = pfw.fid(all_generated_images, all_clean_images)
except ValueError:
    fid_full_genegated = -1
try:
    fid_segments_generated = pfw.fid(all_attack_images, all_clean_images)
except ValueError:
    fid_segments_generated = -1

is_full_generated = inception_score(all_generated_images, resize=True, batch_size=1, cuda=False)[0]
is_segments_generated = inception_score(all_attack_images, resize=True, batch_size=1, cuda=False)[0]

print()
print()
print()
print(f'Original_images_mAP: {mean_ap_clean_total.avg}')
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

results.to_csv('success_attack_is_fid500_val.csv')