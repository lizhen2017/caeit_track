import numpy as np

def xyxy2xywh(bboxes):
    assert bboxes.shape[1] in [4, 5]
    if bboxes.shape[1] == 5:
        bboxes, scores = bboxes[:, :-1], bboxes[:, -1:]
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes