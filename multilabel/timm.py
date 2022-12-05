import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ModelInterface(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 pretrained=False,
                 **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.pretrained = pretrained


    @staticmethod
    def _glob_feature_vector(x, mode, reduce_dims=True):
        if mode == 'avg':
            out = F.adaptive_avg_pool2d(x, 1)
        elif mode == 'max':
            out = F.adaptive_max_pool2d(x, 1)
        elif mode == 'avg+max':
            avg_pool = F.adaptive_avg_pool2d(x, 1)
            max_pool = F.adaptive_max_pool2d(x, 1)
            out = avg_pool + max_pool
        else:
            raise ValueError(f'Unknown pooling mode: {mode}')

        if reduce_dims:
            return out.view(x.size(0), -1)
        return out


class TimmModelsWrapper(ModelInterface):
    def __init__(self,
                 model_name,
                 pretrained=False,
                 dropout_cls = 0.1,
                 pooling_type='avg',
                 extra_head_dim=-1,
                 **kwargs):
        super().__init__(**kwargs)
        self.pretrained = pretrained
        self.is_mobilenet = True if model_name in ["mobilenetv3_large_100_miil_in21k", "mobilenetv3_large_100_miil"] else False
        self.model = timm.create_model(model_name,
                                       pretrained=pretrained,
                                       num_classes=self.num_classes)
        self.num_head_features = self.model.num_features
        self.num_features = (self.model.conv_head.in_channels if self.is_mobilenet
                             else self.model.num_features)
        self.dropout = Dropout(dropout_cls)
        self.pooling_type = pooling_type
        self.model.classifier = self.model.get_classifier()
        if extra_head_dim > 0:
            self.extra_head = nn.Linear(self.num_features, out_features=extra_head_dim)
        else:
            self.extra_head = None

    def forward(self, x):
        y = self.extract_features(x)
        glob_features = self._glob_feature_vector(y, self.pooling_type, reduce_dims=False)
        logits = self.infer_head(glob_features)
        if not self.training:
            return logits
        if self.extra_head is not None:
            extra_features = self.extra_head(glob_features.view(glob_features.shape[0], -1))
            return logits, extra_features
        return (logits,)

    def extract_features(self, x):
        return self.model.forward_features(x)

    def infer_head(self, x):
        if self.is_mobilenet:
            x  = self.model.act2(self.model.conv_head(x))
        self.dropout(x)
        return self.model.classifier(x.view(x.shape[0], -1))

    def get_config_optim(self, lrs):
        parameters = [
            {'params': self.model.named_parameters()},
        ]
        if isinstance(lrs, list):
            assert len(lrs) == len(parameters)
            for lr, param_dict in zip(lrs, parameters):
                param_dict['lr'] = lr
        else:
            assert isinstance(lrs, float)
            for param_dict in parameters:
                param_dict['lr'] = lrs

        return parameters


class Dropout(nn.Module):
    DISTRIBUTIONS = ['none', 'bernoulli', 'gaussian', 'infodrop']

    def __init__(self, p=0.1, mu=0.1, sigma=0.03, dist='bernoulli', kernel=3, temperature=0.2):
        super().__init__()

        self.dist = dist
        assert self.dist in Dropout.DISTRIBUTIONS

        self.p = float(p)
        assert 0. <= self.p <= 1.

        self.mu = float(mu)
        self.sigma = float(sigma)
        assert self.sigma > 0.

        self.kernel = kernel
        assert self.kernel >= 3
        self.temperature = temperature
        assert self.temperature > 0.0

    def forward(self, x, x_original=None):
        if not self.training:
            return x

        if self.dist == 'bernoulli':
            out = F.dropout(x, self.p, self.training)
        elif self.dist == 'gaussian':
            with torch.no_grad():
                soft_mask = x.new_empty(x.size()).normal_(self.mu, self.sigma).clamp_(0., 1.)

            scale = 1. / self.mu
            out = scale * soft_mask * x
        elif self.dist == 'infodrop':
            assert x_original is not None

            out = info_dropout(x_original, self.kernel, x, self.p, self.temperature)
        else:
            out = x

        return out