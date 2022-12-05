import torch
import torch.nn as nn


def bt_loss(x, y, penalty=1.0 / 128.0):
    batch_size = x.shape[0]
    dimensionality = x.shape[1]

    # Barlow Twins loss: redundancy reduction
    batch_norm = nn.BatchNorm1d(dimensionality, affine=False, track_running_stats=False)
    # empirical cross-correlation matrix
    eccm = batch_norm(x).T @ batch_norm(y)
    eccm.div_(batch_size)

    # Compute the invariance term (diagonal) and redundacy term (off-diagonal)
    on_diag = torch.diagonal(eccm).add(-1).pow_(2).sum()
    off_diag = off_diagonal(eccm).pow_(2).sum()
    # Normalize the loss by the dimensionality of the projector
    return (on_diag + penalty * off_diag) / dimensionality


def off_diagonal(x):
    """
    return a tensor containing all the elements outside the diagonal of x
    """
    assert x.shape[0] == x.shape[1]
    return x.flatten()[:-1].view(x.shape[0] - 1, x.shape[0] + 1)[:, 1:].flatten()