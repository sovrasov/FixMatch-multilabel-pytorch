import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from .bt_loss import off_diagonal


class InfoNCELoss(nn.Module):
    def __init__(self, use_gpu=True, s=30, end_s=None, duration_s=None, skip_steps_s=None):
        super(InfoNCELoss, self).__init__()
        self.use_gpu = use_gpu

        assert s > 0
        self.start_s = s
        assert self.start_s > 0.0
        self.end_s = end_s
        self.duration_s = duration_s
        self.skip_steps_s = skip_steps_s
        self.last_scale = self.start_s

    @staticmethod
    def get_last_info():
        return {}

    def get_last_scale(self):
        return self.last_scale

    @staticmethod
    def _get_scale(start_scale, end_scale, duration, skip_steps, iteration, power=1.2):
        def _invalid(_v):
            return _v is None or _v <= 0

        if not _invalid(skip_steps) and iteration < skip_steps:
            return start_scale

        if _invalid(iteration) or _invalid(end_scale) or _invalid(duration):
            return start_scale

        skip_steps = skip_steps if not _invalid(skip_steps) else 0
        steps_to_end = duration - skip_steps
        if iteration < duration:
            factor = (end_scale - start_scale) / (1.0 - power)
            var_a = factor / (steps_to_end ** power)
            var_b = -factor * power / float(steps_to_end)

            iteration -= skip_steps
            out_value = var_a * np.power(iteration, power) + var_b * iteration + start_scale
        else:
            out_value = end_scale

        return out_value

    def forward(self, embd_1, embd_2, target=None, iteration=None):
        self.last_scale = self._get_scale(self.start_s, self.end_s, self.duration_s, self.skip_steps_s, iteration)

        embd_1 = F.normalize(embd_1, p=2, dim=1)
        embd_2 = F.normalize(embd_2, p=2, dim=1)
        num_samples = embd_1.size(0)
        assert num_samples == embd_2.size(0)

        similarities = torch.mm(embd_1, torch.t(embd_2)).clamp(-1, 1)
        all_scores = torch.exp(self.last_scale * similarities)

        pos_scores = torch.diagonal(all_scores)
        neg_scores = off_diagonal(all_scores).view(num_samples, num_samples - 1)

        losses = torch.log(pos_scores / (pos_scores + neg_scores.sum(dim=1))).neg()

        return losses.mean()