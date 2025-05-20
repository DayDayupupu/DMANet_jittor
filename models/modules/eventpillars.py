"""
PointPillars fork from SECOND.
Code written by Alex Lang and Oscar Beijbom, 2018.
Licensed under MIT License [see LICENSE].
"""

import jittor as jt
import jittor.nn as nn


def matrix_multiply(matrix1, matrix2):
    return jt.matmul(matrix1, matrix2)



class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        """
        Pillar Feature Net Layer.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer. This layer performs a similar role as second.pytorch.voxelnet.VFELayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """
        super(PFNLayer,self).__init__()
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels
        self.in_channels = in_channels

        self.linear = nn.Linear(self.in_channels, self.units, bias=False)
        self.norm = nn.BatchNorm2d(self.units, eps=1e-3, momentum=0.01)

        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.units, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(8, 8, kernel_size=(1, 4), stride=(1, 2))

    def execute(self, input):

        x = self.conv1(input)
        x = self.norm(x)
        x = jt.nn.relu(x)
        x = self.conv3(x)
        return x


class PillarFeatureNet(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(64,),
                 with_distance=False):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """

        super(PillarFeatureNet, self).__init__()
        self.name = 'PillarFeatureNet'
        assert len(num_filters) > 0
        num_input_features += 3
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        # Create PillarFeatureNet layers
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(PFNLayer(in_filters, out_filters, use_norm, last_layer=last_layer))
        self.pfn_layers = nn.ModuleList(pfn_layers)

    def execute(self, pillar_x, pillar_y, pillar_z, num_voxels, mask):



        # Find distance of x, y, and z from cluster center
        pillar_xyz = jt.concat((pillar_x, pillar_y, pillar_z), 1)
        # print("Concatenated pillar shape:", pillar_xyz.shape)
        # 计算均值
        points_mean = pillar_xyz.sum(dim=3, keepdims=True) / num_voxels.view(1, 1, -1, 1)
        # print("Points mean shape:", points_mean.shape)

        f_cluster = pillar_xyz - points_mean
        # print("F cluster shape:", f_cluster.shape)

        features_list = [pillar_xyz, f_cluster]
        features = jt.concat(features_list, dim=1)
        masked_features = features * mask
        pillar_feature = self.pfn_layers[0](masked_features)
        return pillar_feature


class EventPillarsScatter(nn.Module):
    def __init__(self,
                 output_shape,
                 num_input_features=64):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """

        super().__init__()
        self.name = 'EventPillarsScatter'
        self.output_shape = output_shape
        self.ny = output_shape[2]
        self.nx = output_shape[3]
        self.nchannels = num_input_features

    # @amp.float_function
    def execute(self, voxel_features, coords):
        # batch_canvas will be the final output.
        batch_canvas = []
        # 组合为一个形状 (channels, height * width)
        shape = (self.nchannels, self.nx * self.ny)
        canvas = jt.zeros(shape, dtype=voxel_features.dtype)
        indices = coords[:, 1] * self.nx + coords[:, 2]
        indices = indices.astype(jt.float64)
        transposed_voxel_features = voxel_features.t()
        # Now scatter the blob back to the canvas.
        indices_2d = indices.view(1, -1)
        ones = jt.ones([self.nchannels, 1], dtype="float64")
        # indices_num_channel = torch.mm(ones, indices_2d)
        indices_num_channel = ones * indices_2d
        indices_num_channel = indices_num_channel.astype(jt.int64)
        scattered_canvas = canvas.scatter_(1, indices_num_channel, transposed_voxel_features)

        # Append to a list for later stacking.
        batch_canvas.append(scattered_canvas)

        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        batch_canvas = jt.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(1, self.nchannels, self.ny, self.nx)
        # print(batch_canvas.shape)
        return batch_canvas