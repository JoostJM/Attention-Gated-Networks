import os
import numpy as np
import utils.util as util
from collections import OrderedDict

import torch
from torch.autograd import Variable
from .base_model import BaseModel
from .layers.loss import *
from .networks_other import benchmark_fp_bp_time
from .utils import classification_stats, get_criterion
from .networks.utils import HookBasedFeatureExtractor


class FeedForwardClassifier(BaseModel):

  def name(self):
    return 'FeedForwardClassifier'

  def __init__(self, experiment, **model_opts):
    super(FeedForwardClassifier, self).__init__(experiment, **model_opts)

    self.labels = None

    # for accumulator
    self.reset_results()

  def get_criterion(self, **train_opts):
      return get_criterion(train_opts['criterion'], 'classifier', self)

  def forward(self, split):
    if split == 'train':
      self.prediction = self.net(Variable(self.input))
    elif split in ['validation', 'test']:
      self.prediction = self.net(Variable(self.input, volatile=True))
      # Apply a softmax and return a segmentation map
      self.logits = self.net.apply_argmax_softmax(self.prediction)
      self.pred = self.logits.data.max(1)

  def test(self):
    self.net.eval()
    self.forward(split='test')
    self.pred = self.compute_logits().data.max(1)
    self.accumulate_results()

  def reset_results(self):
    self.losses = []
    self.pr_lbls = []
    self.pr_probs = []
    self.gt_lbls = []

  def accumulate_results(self):
    self.losses.append(self.loss.data[0])
    self.pr_probs.append(self.pred[0].cpu().numpy())
    self.pr_lbls.append(self.pred[1].cpu().numpy())
    self.gt_lbls.append(self.target.data.cpu().numpy())

  def get_classification_stats(self):
    self.pr_lbls = np.concatenate(self.pr_lbls)
    self.gt_lbls = np.concatenate(self.gt_lbls)
    res = classification_stats(self.pr_lbls, self.gt_lbls, self.labels)
    (self.accuracy, self.f1_micro, self.precision_micro,
     self.recall_micro, self.f1_macro, self.precision_macro,
     self.recall_macro, self.confusion, self.class_accuracies,
     self.f1s, self.precisions, self.recalls) = res

    breakdown = dict(type='table',
                     colnames=['|accuracy|', ' precison|', ' recall|', ' f1_score|'],
                     rownames=self.labels,
                     data=[self.class_accuracies, self.precisions, self.recalls, self.f1s])

    return OrderedDict([('accuracy', self.accuracy),
                        ('confusion', self.confusion),
                        ('f1', self.f1_macro),
                        ('precision', self.precision_macro),
                        ('recall', self.recall_macro),
                        ('confusion', self.confusion),
                        ('breakdown', breakdown)])

  def get_current_errors(self):
    return OrderedDict([('CE', self.loss.data[0])])

  def get_accumulated_errors(self):
    return OrderedDict([('CE', np.mean(self.losses))])

  def get_current_visuals(self):
    inp_img = util.tensor2im(self.input, 'img')
    return OrderedDict([('inp_S', inp_img)])

  def get_feature_maps(self, layer_name, upscale):
    feature_extractor = HookBasedFeatureExtractor(self.net, layer_name, upscale)
    return feature_extractor.forward(Variable(self.input))

  def set_labels(self, labels):
    self.labels = labels

  def load_network_from_path(self, network, network_filepath, strict):
    network_label = os.path.basename(network_filepath)
    epoch_label = network_label.split('_')[0]
    print('Loading the model {0} - epoch {1}'.format(network_label, epoch_label))
    network.load_state_dict(torch.load(network_filepath), strict=strict)
