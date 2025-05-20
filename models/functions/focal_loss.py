import jittor as jt
import jittor.nn as nn


class FocalLoss(nn.Module):
    #def __init__(self):

    def execute(self, classifications, regressions, anchors, annotations):
        # print(annotations)
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []
        # print(anchors)
        anchor = anchors[0, :, :]

        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            classification = jt.clamp(classification, min_v=1e-4, max_v=1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:
                alpha_factor = jt.ones(classification.shape) * alpha
                alpha_factor = 1. - alpha_factor
                focal_weight = classification
                focal_weight = alpha_factor * jt.pow(focal_weight, gamma)

                bce = -(jt.log(1.0 - classification))

                # cls_loss = focal_weight * jt.pow(bce, gamma)
                cls_loss = jt.multiply(focal_weight, bce)
                classification_losses.append(cls_loss.sum())
                regression_losses.append(jt.array(0.0))
                continue

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4])  # num_anchors x num_annotations

            # 修改 max 操作
            IoU_argmax, IoU_max = jt.argmax(IoU, dim=1)  # 使用argmax获取索引

            # compute the loss for classification
            targets = jt.ones(classification.shape) * -1

            targets[IoU_max < 0.4, :] = 0

            positive_indices = IoU_max >= 0.5

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            alpha_factor = jt.ones(targets.shape) * alpha

            alpha_factor = jt.where(targets == 1., alpha_factor, 1. - alpha_factor)
            focal_weight = jt.where(targets == 1., 1. - classification, classification)
            focal_weight = jt.multiply(alpha_factor, jt.pow(focal_weight, gamma))

            bce = -(targets * jt.log(classification) + (1.0 - targets) * jt.log(1.0 - classification))

            # cls_loss = focal_weight * jt.pow(bce, gamma)
            cls_loss = focal_weight * bce
            # print(cls_loss)

            cls_loss = jt.where(targets != -1.0, cls_loss, jt.zeros(cls_loss.shape))
            classification_losses.append(cls_loss.sum() / jt.clamp(num_positive_anchors.float(), min_v=1.0))

            # compute the loss for regression

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths = jt.clamp(gt_widths, min_v=1)
                gt_heights = jt.clamp(gt_heights, min_v=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = jt.log(gt_widths / anchor_widths_pi)
                targets_dh = jt.log(gt_heights / anchor_heights_pi)

                targets = jt.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                targets = targets / jt.array([[0.1, 0.1, 0.2, 0.2]], dtype=jt.float32)


                regression_diff = jt.abs(targets - regression[positive_indices, :])

                # smooth l1 loss
                regression_loss = jt.where(regression_diff <= 1.0 / 9.0,
                                           0.5 * 9.0 * jt.pow(regression_diff, 2),
                                           regression_diff - 0.5 / 9.0)

                # print(regression_loss)
                regression_losses.append(regression_loss.mean())
            else:
                regression_losses.append(jt.array(0.0))


        return jt.stack(classification_losses).mean(dim=0, keepdims=True),jt.stack(regression_losses).mean(dim=0, keepdims=True)


def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = jt.minimum(a[:, 2].unsqueeze(1), b[:, 2]) - jt.maximum(a[:, 0].unsqueeze(1), b[:, 0])
    ih = jt.minimum(a[:, 3].unsqueeze(1), b[:, 3]) - jt.maximum(a[:, 1].unsqueeze(1), b[:, 1])

    iw = jt.clamp(iw, min_v=0)
    ih = jt.clamp(ih, min_v=0)

    ua = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])).unsqueeze(1) + area - iw * ih

    ua = jt.clamp(ua, min_v=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU
