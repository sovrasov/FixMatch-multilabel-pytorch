import numpy as np
import torch


@torch.no_grad()
def accuracy_multilabel(output, target, threshold=0.5):
    batch_size = max(1, target.size(0))

    if isinstance(output, (tuple, list)):
        output = output[0]
    output = torch.sigmoid(output)

    pred_idx = output > threshold
    num_correct = (pred_idx == target).sum(dim=-1)
    num_correct = (num_correct == target.size(1)).sum()

    return num_correct / batch_size


def mAP(targs, preds, pos_thr=0.5):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """
    def average_precision(output, target):
        epsilon = 1e-8

        # sort examples
        indices = output.argsort()[::-1]
        # Computes prec@i
        total_count_ = np.cumsum(np.ones((len(output), 1)))

        target_ = target[indices]
        ind = target_ == 1
        pos_count_ = np.cumsum(ind)
        total = pos_count_[-1]
        pos_count_[np.logical_not(ind)] = 0
        pp = pos_count_ / total_count_
        precision_at_i_ = np.sum(pp)
        precision_at_i = precision_at_i_ / (total + epsilon)

        return precision_at_i

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        scores = preds[:, k]
        targets = targs[:, k]
        ap[k] = average_precision(scores, targets)
    tp, fp, fn, tn = [], [], [], []
    for k in range(preds.shape[0]):
        scores = preds[k,:]
        targets = targs[k,:]
        pred = (scores > pos_thr).astype(np.int32)
        tp.append(((pred + targets) == 2).sum())
        fp.append(((pred - targets) == 1).sum())
        fn.append(((pred - targets) == -1).sum())
        tn.append(((pred + targets) == 0).sum())

    p_c = [tp[i] / (tp[i] + fp[i]) if tp[i] > 0 else 0.0 for i in range(len(tp))]
    r_c = [tp[i] / (tp[i] + fn[i]) if tp[i] > 0 else 0.0
                for i in range(len(tp))]
    f_c = [2 * p_c[i] * r_c[i] / (p_c[i] + r_c[i]) if tp[i] > 0 else 0.0
                for i in range(len(tp))]

    mean_p_c = sum(p_c) / len(p_c)
    mean_r_c = sum(r_c) / len(r_c)
    mean_f_c = sum(f_c) / len(f_c)

    p_o = sum(tp) / (np.array(tp) + np.array(fp)).sum()
    r_o = sum(tp) / (np.array(tp) + np.array(fn)).sum()
    f_o = 2 * p_o * r_o / (p_o + r_o)

    return ap.mean(), mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o