import jittor.nn as nn
import jittor as jt
import numpy as np


class BBoxTransform(nn.Module):
    """
    :param mean: 
    :param std:
    Note that [mean, std] are invariant, in order to enlarge relative values, which will help to regression.
    """

    def __init__(self, mean=None, std=None):
        super(BBoxTransform,self).__init__()
        if mean is None:
            self.mean = jt.array(np.array([0, 0, 0, 0]).astype(np.float32))
        else:
            self.mean = mean
        if std is None:
            self.std = jt.array(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))
        else:
            self.std = std

    def execute(self, boxes, deltas):
        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x = boxes[:, :, 0] + 0.5 * widths
        ctr_y = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = jt.exp(dw) * widths
        pred_h = jt.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = jt.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)
        return pred_boxes


class ClipBoxes(nn.Module):

    def __init__(self, width=None, height=None):
        super(ClipBoxes,self).__init__()

    def execute(self, boxes, img):
        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = jt.clamp(boxes[:, :, 0], min_v=0)
        boxes[:, :, 1] = jt.clamp(boxes[:, :, 1], min_v=0)
        boxes[:, :, 2] = jt.clamp(boxes[:, :, 2], max_v=width)
        boxes[:, :, 3] = jt.clamp(boxes[:, :, 3], max_v=height)

        return boxes


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.transpose())
    area2 = box_area(box2.transpose())

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (jt.minimum(box1[:, None, 2:], box2[:, 2:]) - jt.maximum(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)

    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)
