from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
from PIL import Image
from torchvision.transforms import ToTensor, Normalize
import cv2
import torch
import numpy as np
import blobfile as bf

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

def det_output2numpy(det_results):
    det_results_numpy = []
    for res in det_results:
        if isinstance(res, np.ndarray):
            det_results_numpy.append(res)
        else:
            det_results_numpy.append(res.cpu().detach().numpy())

    return det_results_numpy


# Specify the path to model config and checkpoint file
config_file = '/home/d-d-sh/adv_seg_gan/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = '/home/d-d-sh/adv_seg_gan/PITI/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
path_img = '/home/d-d-sh/datasets/COCO/images/val2017/000000000139.jpg'  # or img = mmcv.imread(img), which will only load it once
img_tensor = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(ToTensor()(Image.open(path_img)))

result = inference_detector(model, denormalize(img_tensor).cuda())
# visualize the results in a new window
# model.show_result(img_tensor, result)
# or save the visualization results to image files
result_numpy = det_output2numpy(result)
det_image = show_result_pyplot(
    model,
    tensor2im(img_tensor),
    result_numpy,
    show=False,
)
print(det_image.shape)
# print([[[int(res[j]) for j in range(4)] for res in result_numpy[i]] for i in range(len(result_numpy))])

cv2.imwrite(f'test_det_coco.jpg', det_image[:, :, ::-1])

path_ann = '/home/d-d-sh/datasets/COCO/annotations/train2017/000000000009.png'

img_tensor = ToTensor()(Image.open(path_ann))
img_numpy = np.asarray(Image.open(path_ann))

print(img_numpy.shape)
img_numpy = np.transpose(img_numpy, (2, 0, 1))
cv2.imwrite(f'mask_test.png', img_numpy[0])
# classes = np.unique(img_numpy)
# masks = []
# for i, cl in enumerate(classes):
#     mask = np.where(img_numpy == cl, 255, 0)
#     mask = np.expand_dims(mask, axis=(0, 3))
#     masks.append(mask)
# masks = np.vstack(masks)
# areas = np.sum(masks, axis=(1, 2, 3))
# top_k = 3
# idx_k = np.flip(np.argsort(areas))[:top_k]
# print(idx_k)

with bf.BlobFile(path_ann, "rb") as f:
    pil_image3 = Image.open(f)
    pil_image3.load()
    pil_image3 = pil_image3.convert("RGB")

label_black = np.asarray(pil_image3)
# print(label_black)
print(label_black.shape)
