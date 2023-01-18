from abc import ABC, abstractmethod

import torch
from utils import AverageMeter


class EMAMeter:
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.val = 0

    def update(self, val):
        self.val = self.alpha * self.val + (1 - self.alpha) * val


class ILossBalancer(torch.nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def balance_losses(self, losses):
        pass

    @abstractmethod
    def init_iteration(self):
        pass

    @abstractmethod
    def end_iteration(self):
        pass


class EqualLossBalancer(ILossBalancer):
    def __init__(self, num_losses) -> None:
        super().__init__()
        self.loss_weights = torch.nn.Parameter(torch.zeros(num_losses, dtype=torch.float32))
        self.optimizer = torch.optim.SGD([self.loss_weights], lr=0.5)

    def forward(self, x):
        pass

    def balance_losses(self, losses):
        total_loss = 0.
        for i, l in enumerate(losses):
            total_loss += torch.exp(-self.loss_weights[i])*l + \
                0.5*self.loss_weights[i]

        return total_loss

    def init_iteration(self):
        self.optimizer.zero_grad()

    def end_iteration(self):
        self.optimizer.step()


class MeanLossBalancer(ILossBalancer):
    def __init__(self, num_losses, weights=None, mode='avg') -> None:
        super().__init__()
        assert mode in ['avg', 'ema']
        if mode == 'avg':
            self.avg_estimators = [AverageMeter() for _ in range(num_losses)]
        else:
            self.avg_estimators = [EMAMeter(0.7) for _ in range(num_losses)]

        if weights is not None:
            assert len(weights) == num_losses
            self.final_weights = weights
        else:
            self.final_weights = [1.] * num_losses

    def forward(self, x):
        pass

    def balance_losses(self, losses):
        total_loss = 0.
        for i, l in enumerate(losses):
            self.avg_estimators[i].update(float(l))
            total_loss += self.final_weights[i] * l / (self.avg_estimators[i].val + 1e-9) * self.avg_estimators[0].val

        return total_loss

    def init_iteration(self):
        pass

    def end_iteration(self):
        pass