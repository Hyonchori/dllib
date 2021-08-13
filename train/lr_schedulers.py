
import torch.optim as optim
import math
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (
                        1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from dllib.models.builder.backbone import BuildBackbone
    cfg = "../models/cfgs/base_backbone_m.yaml"
    bb = BuildBackbone(cfg, info=True)

    opt1 = optim.SGD(params=bb.parameters(), lr=0.001)
    lambdalr = optim.lr_scheduler.LambdaLR(opt1, lr_lambda=lambda epoch: 0.95 ** epoch,
                                    last_epoch=-1)

    opt2 = optim.SGD(params=bb.parameters(), lr=0.001)
    steplr = optim.lr_scheduler.StepLR(opt2, step_size=10, gamma=0.95,
                                       last_epoch=-1)

    opt3 = optim.SGD(params=bb.parameters(), lr=0.001)
    mslr = optim.lr_scheduler.MultiStepLR(opt3, [10, 30, 70, 300], gamma=0.95)

    opt4 = optim.SGD(params=bb.parameters(), lr=0.001)
    coslr = optim.lr_scheduler.CosineAnnealingLR(opt4, T_max=100, eta_min=0)


    opt5 = optim.SGD(params=bb.parameters(), lr=0.001)
    cyclelr = optim.lr_scheduler.CyclicLR(opt5, base_lr=0.0, max_lr=0.001,
                                          step_size_up=60, step_size_down=20,
                                          mode="triangular2")

    opt6 = optim.SGD(params=bb.parameters(), lr=0.001)
    onelr = optim.lr_scheduler.OneCycleLR(opt6, max_lr=0.001, total_steps=500,
                                          anneal_strategy="cos")

    opt7 = optim.SGD(params=bb.parameters(), lr=0.001)
    coswrlr = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt7,
                                                             T_0=100,
                                                             T_mult=2,
                                                             eta_min=0)

    opt8 = optim.SGD(params=bb.parameters(), lr=0.0)
    custom = CosineAnnealingWarmUpRestarts(opt8,
                                           T_0=100,
                                           T_mult=2,
                                           eta_max=0.001,
                                           T_up=10,   # initial warm-up epochs
                                           gamma=0.7)

    lambdalr_lr = []
    steplr_lr = []
    mslr_lr = []
    coslr_lr = []
    cyclelr_lr = []
    onelr_lr = []
    coswrlr_lr = []
    custom_lr = []
    for i in range(500):
        lambdalr_lr.append(opt1.param_groups[0]["lr"])
        lambdalr.step()

        steplr_lr.append(opt2.param_groups[0]["lr"])
        steplr.step()

        mslr_lr.append(opt3.param_groups[0]["lr"])
        mslr.step()

        coslr_lr.append(opt4.param_groups[0]["lr"])
        coslr.step()

        cyclelr_lr.append(opt5.param_groups[0]["lr"])
        cyclelr.step()

        onelr_lr.append(opt6.param_groups[0]["lr"])
        onelr.step()

        coswrlr_lr.append(opt7.param_groups[0]["lr"])
        coswrlr.step()

        custom_lr.append(opt8.param_groups[0]["lr"])
        custom.step()
    '''plt.plot(lambdalr_lr)
    plt.plot(steplr_lr)
    plt.plot(mslr_lr)
    plt.plot(coslr_lr)
    plt.plot(cyclelr_lr)
    plt.plot(onelr_lr)'''
    plt.plot(coswrlr_lr)
    plt.plot(custom_lr)
    plt.show()