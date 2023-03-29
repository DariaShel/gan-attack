import torch
import numpy as np
from pathlib import Path
import pandas as pd


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


def get_bboxes_with_probs(model_output, attack_classes, threshold=0.3):
    bboxes = []
    probs = []
    for cls in attack_classes:
        if model_output[cls].shape[0] != 0:
            prob = model_output[cls][:, -1]
            bbox = model_output[cls][:, :-1]
            bbox = bbox[prob > threshold]
            probs.append(prob)
            bboxes.append(bbox)

    return bboxes, probs


def get_bboxes(model_output, threshold=0.3):
    det_bboxes = []
    for cls in model_output:
        if cls.shape[0] == 0:
            det_bboxes.append(cls)
        else:
            prob = cls[:, -1]
            cls_bboxes = cls[:, :-1]
            cls_bboxes = cls_bboxes[prob > threshold]
            det_bboxes.append(cls_bboxes)

    return det_bboxes


def get_max_segment(label_mask, attacked_segments_id):
    mask_overlay = torch.zeros_like(label_mask)
    for name, id in attacked_segments_id.items():
        mask = torch.where(label_mask == id, 1, 0)
        mask_overlay = torch.bitwise_or(mask_overlay.int(), mask.int())

    return mask_overlay


def overlay(orig_img, gen_img, mask_overlay):
    overlay_img = torch.where(mask_overlay.cuda() == 1, gen_img, orig_img.cuda())
    return overlay_img


def get_detector_loss(probs):
    criterion = torch.nn.BCELoss()

    losses = []
    for prob in probs:
        target = torch.full(prob.shape, 0.1)
        losses.append(criterion(prob, target.cuda()))
    return sum(losses) / len(probs)


def det_output2numpy(det_results):
    det_results_numpy = []
    for res in det_results:
        if isinstance(res, np.ndarray):
            det_results_numpy.append(res)
        else:
            det_results_numpy.append(res.cpu().detach().numpy())

    return det_results_numpy


def get_gt_annotation(annotations_dir, labelids_path, detector_classes):
    annotaton_name = str(Path(labelids_path).name).replace(
        "labelIds.png", "polygons.json"
    )
    annotation_path = f"{annotations_dir}/{annotaton_name}"
    print(annotation_path)
    annotation_df = pd.read_json(annotation_path)
    objects = annotation_df["objects"].values
    bboxes = []
    labels = []
    for obj in objects:
        if obj["label"] in detector_classes.keys():
            x1 = min(obj["polygon"], key=lambda p: p[0])[0]
            y1 = min(obj["polygon"], key=lambda p: p[1])[1]
            x2 = max(obj["polygon"], key=lambda p: p[0])[0]
            y2 = max(obj["polygon"], key=lambda p: p[1])[1]

            labels.append(detector_classes[obj["label"]])
            bboxes.append([x1, y1, x2, y2])

    if len(labels) > 0:
        gt_labels = np.array(labels)
        gt_bboxes = np.array(bboxes)

        gt_info = {
            "bboxes": gt_bboxes,
            "labels": gt_labels,
        }

        return [gt_info]
    else:
        return None
