import argparse
import os
import abc
import tqdm
import jittor as jt
import numpy as np
import jittor.nn as nn
import warnings

warnings.filterwarnings("ignore")

import dataloader.dataset
from models.functions.box_utils import box_iou
from models.modules import dmanet_network
from models.modules.dmanet_detector import DMANet_Detector
from dataloader.loader import Loader
from config.settings import Settings
from utils.metrics import ap_per_class

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

jt.flags.use_cuda = 1

class AbstractTrainer(abc.ABC):
    def __init__(self, settings):
        self.settings = settings
        self.model = None
        self.scheduler = None
        self.nr_classes = None  # numbers of classes
        self.test_loader = None
        self.object_classes = None
        self.nr_test_epochs = None
        self.test_file_indexes = None
        self.nr_input_channels = 8

        self.dataset_builder = dataloader.dataset.getDataloader(self.settings.dataset_name)  # Prophesee
        self.dataset_loader = Loader

        self.createDatasets()  # create test dataset

        self.buildModel()

        self.softmax = nn.Softmax(dim=-1)

        # tqdm progress bar
        self.pbar = None

    @abc.abstractmethod
    def buildModel(self):
        """Model is constructed in child class"""
        pass

    def createDatasets(self):
        """
        Creates the validation and the training data based on the lists specified in the config/settings.yaml file.
        """
        test_dataset = self.dataset_builder(self.settings.dataset_path,
                                            self.settings.object_classes,
                                            self.settings.height,
                                            self.settings.width,
                                            mode="testing",
                                            voxel_size=self.settings.voxel_size,
                                            max_num_points=self.settings.max_num_points,
                                            max_voxels=self.settings.max_voxels,
                                            resize=self.settings.resize,
                                            num_bins=self.settings.num_bins)
        self.test_file_indexes = test_dataset.file_index()
        self.nr_test_epochs = test_dataset.nr_samples
        self.nr_classes = test_dataset.nr_classes
        self.object_classes = test_dataset.object_classes



        self.test_loader = self.dataset_loader(test_dataset, mode="testing",
                                                batch_size=self.settings.batch_size,
                                                num_workers=self.settings.num_cpu_workers,
                                                drop_last=True, )
class RetinaNetDetection(AbstractTrainer):
    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        """
        Create boolean mask by actually number of a padded tensor.
        """
        actual_num = jt.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = jt.arange(max_num).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def process_pillar_input(self, events, idx, idy):
        pillar_x = events[idy][idx][0][..., 0].unsqueeze(0).unsqueeze(0)
        pillar_y = events[idy][idx][0][..., 1].unsqueeze(0).unsqueeze(0)
        pillar_t = events[idy][idx][0][..., 2].unsqueeze(0).unsqueeze(0)
        coors = events[idy][idx][1]
        num_points_per_pillar = events[idy][idx][2].unsqueeze(0)
        num_points_per_a_pillar = pillar_x.size()[3]
        mask = self.get_paddings_indicator(num_points_per_pillar, num_points_per_a_pillar, axis=0)
        mask = mask.permute(0, 2, 1).unsqueeze(1).type_as(pillar_x)
        input = [pillar_x, pillar_y, pillar_t,
                 num_points_per_pillar, mask, coors]
        return input

    def buildModel(self):
        """Creates the specified model"""
        if self.settings.depth == 18:
            self.model = dmanet_network.DMANet18(in_channels=self.settings.nr_input_channels,
                                                 num_classes=len(self.settings.object_classes), pretrained=False)
        else:
            raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

        # self.model = jt.nn.DataParallel(self.model)

    def test(self, args):
        self.pbar = tqdm.tqdm(total=self.nr_test_epochs, unit="Batch", unit_scale=True)
        self.model.load_state_dict(jt.load(args.weights)["state_dict"])

        self.model = self.model.eval()
        retinanet_detector = DMANet_Detector(conf_threshold=args.conf_thresh, iou_threshold=args.iou_thresh)

        iouv = jt.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()  # len(iouv)
        seen = 0
        precision, recall, f1_score, m_precision, m_recall, map50, map = 0., 0., 0., 0., 0., 0., 0.
        stats, ap, ap_class = [], [], []

        max_dets = 100  # 设置最大检测框数量
        total_detection_result = np.zeros((0, max_dets, 6))  # [x1,y1,x2,y2,score,class]
        prev_states, prev_features = None, None

        for i_batch, sample_batched in enumerate(self.test_loader):
            detection_result = []  # detection result for computing mAP
            bounding_box, pos_events, neg_events = sample_batched

            with jt.no_grad():
                for idx in range(self.settings.seq_len):
                    pos_input_list, neg_input_list = [], []
                    for idy in range(self.settings.batch_size // self.settings.batch_size):
                        # process positive/negative events
                        pos_input = self.process_pillar_input(pos_events, idx, idy)
                        neg_input = self.process_pillar_input(neg_events, idx, idy)
                        # print(pos_input)
                        pos_input_list.append(pos_input), neg_input_list.append(neg_input)
                    classification, regression, anchors, prev_states, prev_features, pseudo_img = \
                        self.model([pos_input_list, neg_input_list], prev_states=prev_states,
                                   prev_features=prev_features)
                    out = retinanet_detector(classification, regression, anchors, pseudo_img)
                    # print(out)
                    dets_ = out.numpy()
                    dets_[:, :4] /= self.settings.resize
                    dets_[:, :4] *= np.array(
                        [self.settings.width, self.settings.height, self.settings.width, self.settings.height])
                    detection_result.append(out)

                    # 处理检测结果并保持一致的形状
                    padded_dets = np.zeros((1, max_dets, 6))
                    if len(dets_) > 0:
                        n_dets = min(len(dets_), max_dets)
                        padded_dets[0, :n_dets] = dets_[:n_dets]
                    total_detection_result = np.vstack((total_detection_result, padded_dets))

            self.pbar.update(1)

            for si, pred in enumerate(detection_result):
                bbox = bounding_box[bounding_box[:, -1] == si]  # each batch
                np_labels = bbox[bbox[:, -2] != -1.]
                np_labels = np_labels[:, [4, 0, 1, 2, 3]]  # [cls, coords]
                np_labels = np.ascontiguousarray(np_labels, dtype=np.float32)
                labels = jt.array(np_labels)

                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []
                seen += 1

                if len(pred) == 0:
                    if nl:
                        stats.append((jt.zeros((0, niou), dtype=jt.bool), jt.array([], dtype=jt.float32), jt.array([], dtype=jt.float32), tcls))
                    continue
                # predictions
                predn = pred.clone()

                # assign all predictions as incorrect
                correct = jt.zeros((pred.shape[0]), niou, dtype=jt.bool)
                if nl:
                    detected = []  # target indices
                    tcls_tensor = labels[:, 0]
                    # target boxes
                    tbox = labels[:, 1:5]
                    # per target class
                    for cls in jt.unique(tcls_tensor):
                        ti = jt.nonzero(tcls_tensor == cls).view(-1)
                        pi = jt.nonzero(pred[:, 5] == cls).view(-1)
                        # print(ti,pi)
                        # search for detections
                        if pi.shape[0]:
                            # prediction to target ious
                            i, ious = box_iou(predn[pi, :4], tbox[ti]).argmax(1)  # best ious, indices
                            # append detections
                            detected_set = set()
                            for j in jt.nonzero((ious > iouv[0])):
                                d = ti[i[j]]  # detected target
                                # print(d)
                                if d.item() not in detected_set:
                                    detected_set.add(d.item())
                                    detected.append(d)
                                    correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                    if len(detected) == nl:  # all targets already located in images
                                        break
                stats.append((correct, pred[:, 4], pred[:, 5], tcls))
        self.pbar.close()

        # Directories
        save_dir = os.path.join(self.settings.save_dir, "det_result")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)  # make dir

        # 保存检测结果
        if args.save_npy:
            npy_file = os.path.join(save_dir, "prediction.npy")
            np.save(npy_file, total_detection_result)

        names = {k: v for k, v in enumerate(self.object_classes, start=0)}  # {0: 'Pedestrian', ...}
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            precision, recall, ap, f1_score, ap_class = ap_per_class(*stats, plot=True, save_dir=save_dir, names=names)
            ap50, ap = ap[:, 0], ap.mean(1)
            m_precision, m_recall, map50, map = precision.mean(), recall.mean(), ap50.mean(), ap.mean()
            # nt = np.bincount(stats[3].astype(np.int64), minlength=self.nr_classes)  # number of targets per class
            # print(stats[3])
            nt = np.bincount(stats[3].astype(np.int64), minlength=self.nr_classes)  # number of targets per class
        else:
            nt = jt.zeros(1)

        # Print results
        pf = "%8s" + "%18i" * 2 + "%19.3g" * 4  # print format
        print("\033[0;31m    Class            Events              Labels           Precision           Recall          "
              "   mAP@0.5           mAP@0.5:0.95 \033[0m")
        print(pf % ("all", seen, nt.sum(), m_precision, m_recall, map50, map))

        # Print results per class
        if len(stats):
            for i, c in enumerate(ap_class):
                print(pf % (names[c], seen, nt[c], precision[i], recall[i], ap50[i], ap[i]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test network.")
    parser.add_argument("--settings_file", type=str, default="./config/settings.yaml",
                        help="Path to settings yaml")
    parser.add_argument("--weights", type=str,
                        default="E:\\guobiao\\jittor\\models_path\\model_step_15.pkl",
                        help="model.pth path(s)")
    parser.add_argument("--conf_thresh", type=float, default=0.5,
                        help="object confidence threshold")
    parser.add_argument("--iou_thresh", type=float, default=0.4,
                        help="IoU threshold for NMS")
    parser.add_argument("--save_npy", type=bool, default=True,
                        help="save detection results(predicted bounding boxes), .npy file for visualization")

    args = parser.parse_args()

    settings = Settings(args.settings_file, generate_log=False)

    if settings.model_name == "dmanet":
        tester = RetinaNetDetection(settings)
    else:
        raise ValueError("Model name %s specified in the settings file is not implemented" % settings.model_name)

    tester.test(args)