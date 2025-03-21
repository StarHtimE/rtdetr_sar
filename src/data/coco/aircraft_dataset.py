"""
HRSID dataset for ship detection.
Based on COCO dataset structure but specialized for single "ship" class.
"""

import torch
import torch.utils.data

import torchvision
torchvision.disable_beta_transforms_warning()

from torchvision import datapoints

from pycocotools import mask as coco_mask

from src.core import register

__all__ = ['AircraftDetection']



@register
class AircraftDetection(torchvision.datasets.CocoDetection):
    __inject__ = ['transforms']
    __share__ = ['remap_aircraft_category']
    
    def __init__(self, img_folder, ann_file, transforms, return_masks=False, remap_aircraft_category=False):
        super(AircraftDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertAircraftPolysToMask(return_masks, remap_aircraft_category)
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.return_masks = return_masks
        self.remap_aircraft_category = remap_aircraft_category

    def __getitem__(self, idx):
        img, target = super(AircraftDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        # ['boxes', 'masks', 'labels']:
        if 'boxes' in target:
            target['boxes'] = datapoints.BoundingBox(
                target['boxes'], 
                format=datapoints.BoundingBoxFormat.XYXY, 
                spatial_size=img.size[::-1])  # h w

        if 'masks' in target:
            target['masks'] = datapoints.Mask(target['masks'])

        if self._transforms is not None:
            img, target = self._transforms(img, target)
            
        return img, target

    def extra_repr(self) -> str:
        s = f' img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n'
        s += f' return_masks: {self.return_masks}\n'
        if hasattr(self, '_transforms') and self._transforms is not None:
            s += f' transforms:\n   {repr(self._transforms)}'

        return s


def convert_aircraft_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertAircraftPolysToMask(object):
    def __init__(self, return_masks=False, remap_aircraft_category=False):
        self.return_masks = return_masks
        self.remap_aircraft_category = remap_aircraft_category

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        # sar_aircraft dataset has 7 categories
        if self.remap_aircraft_category:
            classes = [aircraft_category2label[obj["category_id"]] for obj in anno]
        else:
            classes = [obj["category_id"] for obj in anno]

        classes = torch.tensor(classes, dtype=torch.int64) 

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_aircraft_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(w), int(h)])
        target["size"] = torch.as_tensor([int(w), int(h)])
    
        return image, target


# sar_aircraft dataset has 7 categories
# 'classes': ('Boeing737', 'Boeing787', 'A330', 'A220', 'A320/321', 'ARJ21', 'other')
aircraft_category2name = {
    1: 'Boeing737',
    2: 'Boeing787',
    3: 'A330',
    4: 'A220',
    5: 'A320/321',
    6: 'ARJ21',
    7: 'other'
}

aircraft_category2label = {k: i for i, k in enumerate(aircraft_category2name.keys())}
aircraft_label2category = {v: k for k, v in aircraft_category2label.items()}