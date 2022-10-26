import torch
from torch import nn


class AsymmetricLoss(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=0,
                    probability_margin=0.05, eps=1e-8,
                    label_smooth=0.):
        super().__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.label_smooth = label_smooth
        self.clip = probability_margin
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, inputs, targets, scale=1., aug_index=None, lam=None):
        """"
        Parameters
        ----------
        inputs: input logits
        targets: targets (multi-label binarized vector. Elements < 0 are ignored)
        """
        inputs, targets = filter_input(inputs, targets)
        if inputs.shape[0] == 0:
            return 0.

        targets = label_smoothing(targets, self.label_smooth)
        self.anti_targets = 1 - targets

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(scale * inputs)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic BCE calculation
        self.loss = targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            self.xs_pos = self.xs_pos * targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * targets + self.gamma_neg * self.anti_targets)
            self.loss *= self.asymmetric_w

        # sum reduction over batch
        return - self.loss.sum()


def label_smoothing(targets, smooth_degree):
    if smooth_degree > 0:
        targets = targets * (1 - smooth_degree)
        targets[targets == 0] = smooth_degree
    return targets


def filter_input(inputs, targets):
    valid_idx = targets >= 0
    inputs = inputs[valid_idx]
    targets = targets[valid_idx]

    return inputs, targets