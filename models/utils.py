'''
Misc Utility functions
'''

import os
import numpy as np
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from utils.metrics import segmentation_scores, dice_score_list
from sklearn import metrics
from .layers.loss import *


def get_optimizer(params, **optim_opts):
  opt_alg = optim_opts.get('name', 'sgd').lower()
  if opt_alg == 'sgd':
    optimizer = optim.SGD(params, **optim_opts['parameters'])

  elif opt_alg == 'adam':
    optimizer = optim.Adam(params, **optim_opts['parameters'])
  else:
    raise ValueError('optimizer algorithm %s not recognized!' % opt_alg)

  for group in optimizer.param_groups:
    group['initial_lr'] = optim_opts['parameters']['lr']

  return optimizer


def get_criterion(criterion, model_type, model):
  if criterion == 'cross_entropy':
    if model_type == 'seg':
      criterion = cross_entropy_2D if model.config['tensor_dim'] == '2D' else cross_entropy_3D
    elif 'classifier' in model_type:
      criterion = CrossEntropyLoss()
    else:
      raise ValueError('Model Type %s not recognized' % model_type)
  elif criterion == 'dice_loss':
    criterion = SoftDiceLoss(model.config['output_nc'])
  elif criterion == 'dice_loss_pancreas_only':
    criterion = CustomSoftDiceLoss(model.config['output_nc'], class_ids=[0, 2])

  return criterion


def segmentation_stats(pred_seg, target):
  n_classes = pred_seg.size(1)
  pred_lbls = pred_seg.data.max(1)[1].cpu().numpy()
  gt = np.squeeze(target.data.cpu().numpy(), axis=1)
  gts, preds = [], []
  for gt_, pred_ in zip(gt, pred_lbls):
    gts.append(gt_)
    preds.append(pred_)

  iou = segmentation_scores(gts, preds, n_class=n_classes)
  dice = dice_score_list(gts, preds, n_class=n_classes)

  return iou, dice


def classification_scores(gts, preds, labels):
  accuracy = metrics.accuracy_score(gts, preds)
  class_accuracies = []
  for lab in labels:  # TODO Fix
    class_accuracies.append(metrics.accuracy_score(gts[gts == lab], preds[gts == lab]))
  class_accuracies = np.array(class_accuracies)

  f1_micro = metrics.f1_score(gts, preds, average='micro')
  precision_micro = metrics.precision_score(gts, preds, average='micro')
  recall_micro = metrics.recall_score(gts, preds, average='micro')
  f1_macro = metrics.f1_score(gts, preds, average='macro')
  precision_macro = metrics.precision_score(gts, preds, average='macro')
  recall_macro = metrics.recall_score(gts, preds, average='macro')

  # class wise score
  f1s = metrics.f1_score(gts, preds, average=None)
  precisions = metrics.precision_score(gts, preds, average=None)
  recalls = metrics.recall_score(gts, preds, average=None)

  confusion = metrics.confusion_matrix(gts, preds, labels=labels)

  # TODO confusion matrix, recall, precision
  return accuracy, f1_micro, precision_micro, recall_micro, f1_macro, precision_macro, recall_macro, confusion, class_accuracies, f1s, precisions, recalls


def classification_stats(pred_seg, target, labels):
  return classification_scores(target, pred_seg, labels)
