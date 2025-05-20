"""
Define the Smooth L1 Loss
smooth l1 loss { = 0.5x2    if |x| < 1
               { = |x|-0.5  otherwise

In this file, we define loss as follows:
beta = 0.11
smooth l1 loss = 1/(2*beta) * x2   if |x| < beta
               = |x|-beta/2        otherwise
"""
import jittor as jt
import jittor.nn as nn


class Smooth_L1_Loss(nn.Module):
    def __init__(self, beta, reduction):
        super(Smooth_L1_Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction

    def execute(self, inputs, targets):
        targets = jt.array(targets, dtype=jt.float32)
        targets.stop_grad()

        flag = jt.abs(inputs - targets)

        loss = jt.where(flag < self.beta, 0.5/self.beta*flag**2, flag-0.5*self.beta)

        if self.reduction == "mean":
            loss = jt.mean(loss)
        else:
            loss = jt.sum(loss)

        return loss


if __name__ == "__main__":
    loss_function = Smooth_L1_Loss(beta=0.11, reduction="mean")
    inputs = jt.rand(8, 16)
    targets = jt.rand(8, 16)
    loss = loss_function(inputs=inputs, targets=targets)
    print(loss)
