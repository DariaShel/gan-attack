import copy
import functools
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from .glide_util import sample
from . import dist_util, logger
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
from .vgg import VGG
from .adv import AdversarialLoss
from .resample import LossAwareSampler, UniformSampler
import glob
import torchvision.utils as tvu
import PIL.Image as Image

from mmdet.apis import inference_detector, show_result_pyplot
import cv2

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0
 
 

class TrainLoop:
    def __init__(
        self,
        model,
        glide_options,
        model_kwargs,
        diffusion,
        detector,
        data,
        val_data,
        img_id,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        finetune_decoder = False,
        mode = '',
        use_vgg = False,
        use_gan = False,
        uncond_p = 0,
        super_res = 0,
    ):
        self.model = model
        self.model_kwargs = model_kwargs
        self.detector = detector
        self.img_id = img_id
        self.glide_options=glide_options
        self.diffusion = diffusion
        self.data = data
        self.val_data=val_data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = find_resume_checkpoint(resume_checkpoint)
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        if use_vgg:
            self.vgg = VGG(conv_index='22').to(dist_util.dev())
            print('use perc')
        else:
            self.vgg = None

        if use_gan:
            self.adv = AdversarialLoss()
            print('use adv')
        else:
            self.adv = None

        self.super_res = super_res
         
        self.uncond_p =uncond_p
        self.mode = mode

        self.finetune_decoder = finetune_decoder
        if finetune_decoder:
            self.optimize_model = self.model
        else:          
            self.optimize_model = self.model.encoder
         
        self.model_params = list(self.optimize_model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()
        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(th.load(resume_checkpoint, map_location="cpu"),strict=False)

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = th.load(ema_checkpoint, map_location=dist_util.dev())
                ema_params = self._state_dict_to_master_params(state_dict)

        #dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = th.load(opt_checkpoint, map_location="cpu")
            try:
                self.opt.load_state_dict(state_dict)
            except:
                pass

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def get_mask(self, gt_mask):
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

    def run_loop(self):
        mask_overlay = self.get_mask(self.model_kwargs['mask'][0].numpy())
        while (
            not self.lr_anneal_steps
            or self.step <= self.lr_anneal_steps
        ):

            # uncond_p = 0
            # if self.super_res:
            #     uncond_p = 0
            # elif   self.finetune_decoder:
            #     uncond_p = self.uncond_p
            # elif  self.step > self.lr_anneal_steps - 40000:
            #     uncond_p = self.uncond_p

            self.run_step(mask_overlay)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # self.val(self.step, mask_overlay)
            self.step += 1
         
        if (self.step - 1) % self.save_interval != 0:
            self.save()
        attack_results = self.val(self.step, mask_overlay)

        return attack_results


    def run_step(self, mask_overlay):
        self.forward_backward(mask_overlay)
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()

    def forward_backward(self, mask_overlay):
        zero_grad(self.model_params)
        for i in range(0, self.data.shape[0], self.microbatch):
            micro = self.data[i : i + self.microbatch].to(dist_util.dev())
            micro_cond={n:self.model_kwargs[n][i:i+self.microbatch].to(dist_util.dev()) for n in self.model_kwargs if n  in ['ref', 'low_res', 'mask_overlay']}
            last_batch = (i + self.microbatch) >= self.data.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
             
            if self.step < 100:
                vgg_loss = None
                adv_loss = None
            else:
                vgg_loss = self.vgg
                adv_loss = self.adv
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                self.detector,
                micro,
                t,
                vgg_loss,
                adv_loss,
                mask_overlay,
                model_kwargs=micro_cond,
            )
            
            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()
           
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )
            
            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            if self.use_fp16:
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()

    def val(self, step, mask_overlay):
        inner_model=self.ddp_model.module
        inner_model.eval()
        if dist.get_rank() == 0:
            print("sampling...")   

        s_path = os.path.join(logger.get_dir(), 'results1')
        os.makedirs(s_path,exist_ok=True)
        guidance_scale=self.glide_options['sample_c']
       
              
        with th.no_grad():
            samples=sample(
                glide_model=inner_model,
                glide_options=self.glide_options,
                side_x=self.glide_options['image_size'],
                side_y=self.glide_options['image_size'],
                prompt=self.model_kwargs,
                batch_size=self.glide_options['batch_size'],
                guidance_scale=guidance_scale,
                device=dist_util.dev(),
                prediction_respacing=self.glide_options['sample_respacing'],
                upsample_enabled=self.glide_options['super_res'],
                upsample_temp=0.997,
                mode = self.mode,
            )
            
            samples = samples.cpu()
    
            ref = self.model_kwargs['ref_ori']
            # LR = model_kwargs['low_res'].cpu()

            for i in range(samples.size(0)):
                out_path = os.path.join(s_path, f"{dist.get_rank()}_{self.img_id}_step{step}_{guidance_scale}_output.png")
                tvu.save_image(
                    (samples[i]+1)*0.5, out_path)

                out_path = os.path.join(s_path, f"{dist.get_rank()}_{self.img_id}_step{step}_{guidance_scale}_gt.png")
                tvu.save_image(
                    (self.val_data[i]+1)*0.5, out_path)

                out_path = os.path.join(s_path, f"{dist.get_rank()}_{self.img_id}_step{step}_{guidance_scale}_ref.png")
                tvu.save_image(
                    (ref[i]+1)*0.5, out_path)
                
                out_path = os.path.join(s_path, f"{dist.get_rank()}_{self.img_id}_step{step}_{guidance_scale}_det_attack.png")
                overlay_img = th.where(mask_overlay.to(dist_util.dev()) == 1, samples[i].to(dist_util.dev()), self.val_data[i].to(dist_util.dev()))
                pred_det_attack = inference_detector(self.detector, self.denormalize(overlay_img).to(dist_util.dev()))
                result_numpy_attack = self.det_output2numpy(pred_det_attack)
                det_attack = show_result_pyplot(
                    self.detector,
                    self.tensor2im(overlay_img),
                    result_numpy_attack,
                    show=False,
                )
                cv2.imwrite(out_path, det_attack[:, :, ::-1])

                out_path = os.path.join(s_path, f"{dist.get_rank()}_{self.img_id}_step{step}_{guidance_scale}_attack.png")
                cv2.imwrite(out_path, self.tensor2im(overlay_img)[:, :, ::-1])

                out_path = os.path.join(s_path, f"{dist.get_rank()}_{self.img_id}_step{step}_{guidance_scale}_det_orig.png")
                pred_det_clean = inference_detector(self.detector, self.denormalize(self.val_data[i]).to(dist_util.dev()))
                result_numpy_clean = self.det_output2numpy(pred_det_clean)
                det_orig = show_result_pyplot(
                    self.detector,
                    self.tensor2im(self.val_data[i]),
                    result_numpy_clean,
                    show=False,
                )

                cv2.imwrite(out_path, det_orig[:, :, ::-1])

                # out_path = os.path.join(s_path, f"{dist.get_rank()}_{img_id}_step{step}_{guidance_scale}_lr.png")
                # tvu.save_image(
                #     (LR[i]+1)*0.5, out_path)
    
        inner_model.train()

        return result_numpy_clean, result_numpy_attack, samples[0], overlay_img, self.val_data[0], self.model_kwargs['path'][0]

    def denormalize(self, image_tensor):
        if isinstance(image_tensor, list):
            denorm_image = []
            for i in range(len(image_tensor)):
                denorm_image.append(self.denormalize(image_tensor[i]))
            return denorm_image
        denorm_image = image_tensor.clone()
        
        denorm_image = (denorm_image + 1) / 2.0 * 255.0  
        denorm_image = th.clip(denorm_image, 0, 255)

        return denorm_image

    def tensor2im(self, image_tensor, imtype=np.uint8, normalize=True):
        if isinstance(image_tensor, list):
            image_numpy = []
            for i in range(len(image_tensor)):
                image_numpy.append(self.tensor2im(image_tensor[i], imtype, normalize))
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
    
    def det_output2numpy(self, det_results):
        det_results_numpy = []
        for res in det_results:
            if isinstance(res, np.ndarray):
                det_results_numpy.append(res)
            else:
                det_results_numpy.append(res.cpu().detach().numpy())

        return det_results_numpy


    def optimize_fp16(self):
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
            sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        return
   

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(
                list(self.optimize_model.parameters()), master_params
            )
        state_dict = self.optimize_model.state_dict()
        for i, (name, _value) in enumerate(self.optimize_model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.optimize_model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    filename=filename.split('/')[-1]
    assert(filename.endswith(".pt"))
    filename=filename[:-3]
    if filename.startswith("model"):
        split = filename[5:]
    elif filename.startswith("ema"):
        split = filename.split("_")[-1]
    else:
        return 0
    try:
        return int(split)
    except ValueError:
        return 0


def get_blob_logdir():
    p=os.path.join(logger.get_dir(),"checkpoints")
    os.makedirs(p,exist_ok=True)
    return p

def find_resume_checkpoint(resume_checkpoint):
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    if not resume_checkpoint:
        return None
    if "ROOT" in resume_checkpoint:
        maybe_root=os.environ.get("AMLT_MAP_INPUT_DIR")
        maybe_root="OUTPUT/log" if not maybe_root else maybe_root
        root=os.path.join(maybe_root,"checkpoints")
        resume_checkpoint=resume_checkpoint.replace("ROOT",root)
    if "LATEST" in resume_checkpoint:
        files=glob.glob(resume_checkpoint.replace("LATEST","*.pt"))
        if not files:
            return None
        return max(files,key=parse_resume_step_from_filename)
    return resume_checkpoint



def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

