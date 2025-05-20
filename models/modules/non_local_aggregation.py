import jittor as jt
import jittor.nn as nn



class NonLocalAggregationModule(nn.Module):

    def __init__(self, in_channels, reduction=2):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // reduction

        self.g = nn.Conv(self.in_channels, self.inter_channels, kernel_size=1)
        self.g_max = nn.Pool(kernel_size=2, stride=2, op="maximum")
        self.theta = nn.Conv(self.in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv(self.in_channels, self.inter_channels, kernel_size=1)
        self.phi_max = nn.Pool(kernel_size=2, stride=2, op="maximum")
        self.att_layer = nn.Conv(self.inter_channels, self.in_channels, kernel_size=1)
        # self.conv1 = nn.Sequential(nn.Conv(self.in_channels, self.in_channels, kernel_size=1),
        #                            nn.ReLU(),
        #                            nn.Conv(self.in_channels, self.in_channels, kernel_size=1))
        # self.conv2 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
        #                            nn.Conv(self.in_channels, self.in_channels, kernel_size=1),
        #                            nn.ReLU(),
        #                            nn.Conv(self.in_channels, self.in_channels, kernel_size=1),
        #                            nn.Sigmoid())

        # self.conv1 = nn.Sequential(nn.Conv(self.in_channels, self.in_channels, kernel_size=1),
        #                            nn.ReLU())
        # self.conv2 = nn.Sequential(nn.Conv(self.in_channels, self.in_channels, kernel_size=1),
        #                            nn.ReLU(),
        #                            nn.AdaptiveAvgPool2d((1, 1)),
        #                            nn.Sigmoid())

        self.alpha = 0.1
        self.init_weights()

    def execute(self, curr_x, adja_x):
        """
        :param curr_x: current frame (HxWxC)
        :param adja_x: adjacent frames (NxHxWxC)
        """
        n, _, h, w = adja_x.shape
        g_x = self.g(adja_x)
        g_x = self.g_max(g_x).view(n, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)  # N x HW x C

        theta_x = self.theta(curr_x).view(1, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)  # 1 x HW x C

        phi_x = self.phi(adja_x)
        phi_x = self.phi_max(phi_x).view(n, self.inter_channels, -1)  # N x C x HW
        pairwise_weight = jt.matmul(theta_x, phi_x)  # N x HW x HW

        # pairwise_weight /= theta_x.shape[-1]
        pairwise_weight /= theta_x.shape[-1] ** 0.5
        pairwise_weight = nn.softmax(pairwise_weight, dim=-1)

        y = jt.matmul(pairwise_weight, g_x)  # N x HW x C
        y = y.permute(0, 2, 1).contiguous().reshape(n, self.inter_channels, h, w)  # must contiguous
        att_x = self.att_layer(y)
        x = curr_x + att_x
        # x1 = self.conv1(x)
        # x2 = self.conv2(x1)
        # x_ct = x1 * x2
        # out = x*self.alpha + x_ct
        return x

    def init_weights(self, std=0.01):
        for m in [self.g, self.theta, self.phi]:
            normal_init(m, std=std)
        normal_init(self.att_layer, 0)


def normal_init(module, mean=0, std=1, bias=0):
    nn.init.gauss_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)

