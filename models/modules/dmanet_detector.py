import jittor as jt
import jittor.nn as nn
from models.functions.box_utils import BBoxTransform, ClipBoxes


class DMANet_Detector(nn.Module):
    def __init__(self, conf_threshold, iou_threshold):
        super(DMANet_Detector, self).__init__()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()

    def execute(self, classification, regression, anchors, img_batch):
        transformed_anchors = self.regressBoxes(anchors, regression)
        transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

        final_result = [[], [], []]

        final_scores = jt.array([])
        final_anchor_boxes_indexes = jt.array([]).long()
        final_anchor_boxes_coordinates = jt.array([])

        for i in range(classification.shape[2]):
            # print(classification.shape)
            scores = jt.squeeze(classification[:, :, i])
            scores_over_thresh = scores > self.conf_threshold
            # print(scores)
            if not scores_over_thresh.any():
                continue

            scores = scores[scores_over_thresh]
            anchor_boxes = jt.squeeze(transformed_anchors)
            anchor_boxes = anchor_boxes[scores_over_thresh]
            scores_N = scores.unsqueeze(1)  # [N] → [N, 1]
            anchor_idx = jt.concat([anchor_boxes, scores_N], dim=1)  # [N, 4] + [N, 1] → [N, 5]

            anchors_nms_idx = jt.misc.nms(anchor_idx, self.iou_threshold)

            final_result[0].extend(scores[anchors_nms_idx])
            final_result[1].extend(jt.array([i] * anchors_nms_idx.shape[0], dtype=jt.int32))
            final_result[2].extend(anchor_boxes[anchors_nms_idx])

            final_scores = jt.concat((final_scores, scores[anchors_nms_idx]))

            final_anchor_boxes_indexes_value = jt.array([i] * anchors_nms_idx.shape[0], dtype=jt.int32)

            final_anchor_boxes_indexes = jt.concat((final_anchor_boxes_indexes, final_anchor_boxes_indexes_value))
            final_anchor_boxes_coordinates = jt.concat((final_anchor_boxes_coordinates, anchor_boxes[anchors_nms_idx]))

        if len(final_scores):
            final_scores = jt.unsqueeze(final_scores, dim=1)
            final_anchor_boxes_indexes = jt.unsqueeze(final_anchor_boxes_indexes, dim=1)

            return jt.concat([final_anchor_boxes_coordinates, final_scores, final_anchor_boxes_indexes], dim=1)
        else:
            return jt.array([]).reshape(-1, 6)
