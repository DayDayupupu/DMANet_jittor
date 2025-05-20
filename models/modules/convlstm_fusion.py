import jittor as jt
import jittor.nn as nn
from models.modules.embed_aggregator import EmbedAggregator


class ConvLSTM(nn.Module):
    """Adapted from: https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py """

    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        pad = kernel_size // 2  # 保持大小不变

        # cache a tensor filled with zeros to
        # avoid reallocating memory at each inference step if --no-recurrent is enabled
        self.zero_tensors = {}
        self.embed_layer = EmbedAggregator(channels=hidden_size)
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=pad)

        # 添加激活函数
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def execute(self, input_, prev_state=None):
        flag = prev_state
        # get batch and spatial sizes
        batch_size = input_.shape[0]
        spatial_size = input_.shape[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            # create the zero tensor if it has not been created already
            state_size = tuple([batch_size, self.hidden_size] + list(spatial_size))
            if state_size not in self.zero_tensors:
                # allocate a tensor with size `spatial_size`, filled with zero (if it has not been allocated already)
                self.zero_tensors[state_size] = (
                    jt.zeros(state_size),
                    jt.zeros(state_size),
                    jt.zeros(state_size)
                )

            prev_state = self.zero_tensors[tuple(state_size)]

        prev_hidden, prev_cell, prev_feature = prev_state
        if flag is None:
            fusion_hidden = input_
        else:
            weights = []
            for idx in range(batch_size):
                weight = self.embed_layer(input_[idx].unsqueeze(0), prev_feature[idx].unsqueeze(0))
                weights.append(weight)
            weights = jt.concat([w for w in weights])
            fusion_hidden = prev_hidden * weights
            prev_cell = prev_cell * weights

        # prophesee_dlut size is [batch, channel, height, width]
        stacked_inputs = jt.concat((input_, fusion_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = jt.chunk(gates, 4, 1)

        # apply sigmoid non linearity
        in_gate = self.sigmoid(in_gate)
        remember_gate = self.sigmoid(remember_gate)
        out_gate = self.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = self.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * self.tanh(cell)

        return hidden, cell, input_
