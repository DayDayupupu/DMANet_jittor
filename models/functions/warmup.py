import jittor as jt
import jittor.nn as nn

class WarmUpLR:
    def __init__(self, optimizer, total_iters):
        self.optimizer = optimizer
        self.total_iters = total_iters
        self.last_epoch = 0
        self.base_lr = optimizer.lr

    def step(self):
        self.last_epoch += 1
        lr = self.base_lr * self.last_epoch / (self.total_iters + 1e-8)
        self.optimizer.lr = lr

class ExponentialDecayLR(nn.LRScheduler):
    """指数衰减学习率调度器
    
    参数:
        optimizer (Optimizer): 优化器
        decay_rate (float): 衰减率
        last_epoch (int, optional): 最后的轮次. 默认为 -1
    """
    def __init__(self, optimizer, decay_rate, last_epoch=-1):
        self.decay_rate = decay_rate
        super(ExponentialDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """计算当前学习率
        
        返回:
            float: 当前学习率
        """
        return [base_lr * (self.decay_rate ** self.last_epoch) 
                for base_lr in self.base_lrs]