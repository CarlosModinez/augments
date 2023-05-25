import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from pycocotools.coco import COCO

class CocoDataset(torch.utils.data.Dataset):
  def __init__(self, root, transforms=None):
    dataset_path = os.path.join(root)
    ann_file = os.path.join(dataset_path, "pannuke_train.json")
    self.imgs_dir = os.path.join(dataset_path, "train")
    self.coco = COCO(ann_file)
    self.img_ids = self.coco.getImgIds()
    
    self.transforms = transforms

  def __getitem__(self, idx):
    img_id = self.img_ids[idx]
    img_obj = self.coco.loadImgs(img_id)[0]
    anns_obj = self.coco.loadAnns(self.coco.getAnnIds(img_id)) 

    img = Image.open(os.path.join(self.imgs_dir, img_obj['file_name']))

    bboxes = [ann['bbox'] for ann in anns_obj]
    bboxes = [[ann[0], ann[1], ann[0] + ann[2], ann[1] + ann[3]] for ann in bboxes]
    masks = [self.coco.annToMask(ann) for ann in anns_obj]
    areas = [ann['area'] for ann in anns_obj]
    labels = [ann['category_id'] for ann in anns_obj]

    boxes = torch.as_tensor(bboxes, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.int64)
    masks = torch.as_tensor(masks, dtype=torch.uint8)
    image_id = torch.tensor([idx])
    area = torch.as_tensor(areas)
    iscrowd = torch.zeros(len(anns_obj), dtype=torch.int64)

    target = {}
    target["boxes"] = boxes
    target["labels"] = labels
    target["masks"] = masks
    target["image_id"] = image_id
    target["area"] = area
    target["iscrowd"] = iscrowd

    if self.transforms is not None:
        img, target = self.transforms(img, target)
    return img, target

  def __len__(self):
    return len(self.img_ids)
