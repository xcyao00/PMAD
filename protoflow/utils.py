import os
import math
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

import torch
import torch.nn.functional as F


def min_max_normalize(array):
    max_val = array.max()
    min_val = array.min()
    
    return (array - min_val) / (max_val - min_val)


def thresholding(labels, masks, img_scores, scores):
    precision, recall, thresholds = precision_recall_curve(labels.flatten(), img_scores.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    image_threshold = thresholds[np.argmax(f1)]

    precision, recall, thresholds = precision_recall_curve(masks.flatten(), scores.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    pixel_threshold = thresholds[np.argmax(f1)]

    return image_threshold, pixel_threshold


def measure_rocaucs(labels, masks, img_scores, scores):
    # apply gaussian smoothing on the score map
    for i in range(scores.shape[0]):
        scores[i] = gaussian_filter(scores[i], sigma=4)

    # calculate image-level ROC-AUC score
    image_auc = roc_auc_score(labels, img_scores)
    
    # calculate pixel-level ROC-AUC score
    pixel_auc = roc_auc_score(masks.flatten(), scores.flatten())

    # using f1-score to get threshold
    thresholds = thresholding(labels, masks, img_scores, scores)

    return image_auc, pixel_auc, thresholds


def load_weights(encoder, decoders, filename):
    path = os.path.join(WEIGHT_DIR, filename)
    state = torch.load(path)
    encoder.load_state_dict(state['encoder_state_dict'], strict=False)
    decoders = [decoder.load_state_dict(state, strict=False) for decoder, state in zip(decoders, state['decoder_state_dict'])]
    print('Loading weights from {}'.format(filename))


def adjust_learning_rate(c, optimizer, epoch):
    lr = c.lr
    if c.lr_cosine:
        eta_min = lr * (c.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / c.meta_epochs)) / 2
    else:
        steps = np.sum(epoch >= np.asarray(c.lr_decay_epochs))
        if steps > 0:
            lr = lr * (c.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(c, epoch, batch_id, total_batches, optimizer):
    if c.lr_warm and epoch < c.lr_warm_epochs:
        p = (batch_id + epoch * total_batches) / \
            (c.lr_warm_epochs * total_batches)
        lr = c.lr_warmup_from + p * (c.lr_warmup_to - c.lr_warmup_from)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    #
    for param_group in optimizer.param_groups:
        lrate = param_group['lr']
    return lrate


def embedding_concat(x, y):
    # from https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = x.new_zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z
