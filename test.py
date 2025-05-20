"""
Example usage: CUDA_VISIBLE_DEVICES=1, python test_DMANet.py --settings_file "config/settings.yaml" --test_file "path/to/test/file"
"""
import argparse
import os
import jittor as jt
import numpy as np
import jittor.nn as nn
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

import dataloader.dataset
from models.modules import dmanet_network
from models.modules.dmanet_detector import DMANet_Detector
from dataloader.loader import Loader
from config.settings import Settings
from utils.metrics import ap_per_class
from utils.visualizations import drawBoundingBoxes
import cv2

class DMANetTester:
    def __init__(self, settings):
        jt.flags.use_cuda = 1
        self.settings = settings
        self.model = None
        self.nr_classes = None
        self.test_loader = None
        self.object_classes = None
        self.writer = SummaryWriter(self.settings.ckpt_dir)

        # 初始化数据集和数据加载器
        self.dataset_builder = dataloader.dataset.getDataloader(self.settings.dataset_name)
        self.dataset_loader = Loader
        self.createTestDataset()

        # 构建模型
        self.buildModel()

        # 加载预训练权重
        if self.settings.resume_training:
            self.loadCheckpoint(self.settings.resume_ckpt_file)

    def createTestDataset(self):
        """创建测试数据集"""
        test_dataset = self.dataset_builder(
            self.settings.dataset_path,
            self.settings.object_classes,
            self.settings.height,
            self.settings.width,
            mode="testing",  # 使用测试模式
            voxel_size=self.settings.voxel_size,
            max_num_points=self.settings.max_num_points,
            max_voxels=self.settings.max_voxels,
            resize=self.settings.resize,
            num_bins=self.settings.num_bins
        )

        self.nr_classes = test_dataset.nr_classes
        self.object_classes = test_dataset.object_classes

        self.test_loader = self.dataset_loader(
            test_dataset,
            mode="testing",
            batch_size=1,  # 测试时通常使用batch_size=1
            num_workers=self.settings.num_cpu_workers,
            drop_last=False
        )

    def buildModel(self):
        """构建模型"""
        if self.settings.depth == 18:
            self.model = dmanet_network.DMANet18(
                in_channels=self.settings.nr_input_channels,
                num_classes=len(self.settings.object_classes),
                pretrained=False
            )
        else:
            raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    def loadCheckpoint(self, filename):
        """加载检查点"""
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = jt.load(filename)
            self.model.load_state_dict(checkpoint["state_dict"])
            print("=> loaded checkpoint '{}'".format(filename))
        else:
            print("=> no checkpoint found at '{}'".format(filename))

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        """创建填充指示器"""
        actual_num = jt.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = jt.arange(max_num, dtype=int).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def process_pillar_input(self, events, idx, idy):
        """处理pillar输入"""
        pillar_x = events[idy][idx][0][..., 0].unsqueeze(0).unsqueeze(0)
        pillar_y = events[idy][idx][0][..., 1].unsqueeze(0).unsqueeze(0)
        pillar_t = events[idy][idx][0][..., 2].unsqueeze(0).unsqueeze(0)
        coors = events[idy][idx][1]
        num_points_per_pillar = events[idy][idx][2].unsqueeze(0)
        num_points_per_a_pillar = pillar_x.size()[3]
        mask = self.get_paddings_indicator(num_points_per_pillar, num_points_per_a_pillar, axis=0)
        mask = mask.permute(0, 2, 1).unsqueeze(1).type_as(pillar_x)
        input = [pillar_x, pillar_y, pillar_t, num_points_per_pillar, mask, coors]
        return input

    def test(self):
        """执行测试"""
        self.model.eval()
        dmanet_detector = DMANet_Detector(conf_threshold=0.2, iou_threshold=0.5)

        # 创建保存结果的目录
        save_dir = os.path.join(self.settings.save_dir, "test_results")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with jt.no_grad():
            for i_batch, sample_batched in enumerate(self.test_loader):
                bounding_box, pos_events, neg_events = sample_batched
                prev_states, prev_features = None, None

                for idx in range(self.settings.seq_len):
                    pos_input_list, neg_input_list = [], []
                    for idy in range(1):  # 测试时batch_size=1
                        # 处理正负事件
                        pos_input = self.process_pillar_input(pos_events, idx, idy)
                        neg_input = self.process_pillar_input(neg_events, idx, idy)
                        pos_input_list.append(pos_input)
                        neg_input_list.append(neg_input)

                    # 前向传播
                    classification, regression, anchors, prev_states, prev_features, pseudo_img = \
                        self.model([pos_input_list, neg_input_list], prev_states=prev_states, prev_features=prev_features)

                    # 检测
                    detections = dmanet_detector(classification, regression, anchors, pseudo_img)

                    # 保存检测结果
                    if len(detections) > 0:
                        # 将检测结果转换为图像格式
                        img = pseudo_img[0].numpy().transpose(1, 2, 0)
                        img = (img * 255).astype(np.uint8)

                        # 绘制检测框
                        class_names = [self.object_classes[int(d[5])] for d in detections]
                        img_with_boxes = drawBoundingBoxes(img, detections[:, :4], class_names, ground_truth=False)

                        # 保存结果
                        save_path = os.path.join(save_dir, f"frame_{i_batch}_{idx}.jpg")
                        cv2.imwrite(save_path, img_with_boxes)

                        # 保存检测结果到文本文件
                        txt_path = os.path.join(save_dir, f"frame_{i_batch}_{idx}.txt")
                        with open(txt_path, 'w') as f:
                            for det, cls_name in zip(detections, class_names):
                                f.write(f"{cls_name} {det[4]:.2f} {det[0]:.1f} {det[1]:.1f} {det[2]:.1f} {det[3]:.1f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test network.")
    parser.add_argument("--settings_file", type=str, default="config/settings.yaml",
                        help="Path to settings yaml")
    parser.add_argument("--test_file", type=str, required=True,
                        help="Path to test file")


    # parser.add_argument("--weights", type=str,
    #                     default="./models_path/model_step_0",
    #                     help="model.pth path(s)")
    parser.add_argument("--weights", type=str,
                        default="E:\\guobiao\\jittor\\log\\20250518-213418\\checkpoints\\model_step_30",
                        help="model.pth path(s)")


    parser.add_argument("--save_npy", type=bool, default=True,
                        help="save detection results(predicted bounding boxes), .npy file for visualization")
    parser.add_argument("--debug", type=bool, default=True,  # 添加调试模式
                        help="enable debug mode to print more information")

    args = parser.parse_args()
    settings_filepath = args.settings_file
    settings = Settings(settings_filepath, generate_log=True)

    if settings.model_name == "dmanet":
        tester = DMANetTester(settings)
    else:
        raise ValueError("Model name %s specified in the settings file is not implemented" % settings.model_name)

    tester.test()