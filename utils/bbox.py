

import numpy as np
import torch
from torchvision.ops.boxes import box_iou as th_box_iou

def xyxy2xywh(bboxes):
    assert bboxes.shape[1] in [4, 5]
    if bboxes.shape[1] == 5:
        bboxes, scores = bboxes[:, :-1], bboxes[:, -1:]
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


def xywh2xyxy(bboxes):
    assert bboxes.shape[1] == 4
    bboxes[:, 2] = bboxes[:, 2] + bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] + bboxes[:, 1]
    return bboxes


def box_iou(boxes1, boxes2):
    if isinstance(boxes1, np.ndarray):
        boxes1 = torch.from_numpy(boxes1).float()
    if isinstance(boxes2, np.ndarray):
        boxes2 = torch.from_numpy(boxes2)
    iou_mat = th_box_iou(boxes1, boxes2)
    return iou_mat.numpy()
