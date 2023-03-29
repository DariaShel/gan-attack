from mmdet.datasets.coco import CocoDataset

coco = CocoDataset()

ann = coco.load_annotations('/home/d-d-sh/adv_seg_gan/PITI/logs/coco-upsample/results/0_45_step2001_1.0_ref.png')
print(ann)