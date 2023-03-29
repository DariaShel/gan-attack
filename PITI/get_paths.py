from pathlib import Path

val_imgs_path = Path('/home/d-d-sh/datasets/COCO/images/val2017')
val_ann_color_path = Path('/home/d-d-sh/datasets/COCO/annotations_color/val2017')
val_ann_path = Path('/home/d-d-sh/datasets/COCO/annotations/val2017')

list_val_imgs = list(val_imgs_path.glob('**/*.jpg'))
list_val_ann_color = list(val_ann_color_path.glob('**/*.png'))
list_val_ann = list(val_ann_path.glob('**/*.png'))

list_val_imgs.sort()
list_val_ann.sort()
list_val_ann_color.sort()

with open('dataset/coco_val5.txt', 'w') as f:
    for i, (img, ann, ann_color) in enumerate(zip(list_val_imgs, list_val_ann, list_val_ann_color)):
        if i == 5:
            break
        f.write(f'{img} {ann} {ann_color}\n')
