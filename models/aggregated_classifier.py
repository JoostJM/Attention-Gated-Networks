import collections
import numpy as np
import torch
from torch.autograd import Variable
from .feedforward_classifier import FeedForwardClassifier


class AggregatedClassifier(FeedForwardClassifier):
  def name(self):
    return 'AggregatedClassifier'

  def __init__(self, experiment, **model_opts):
    super(AggregatedClassifier, self).__init__(experiment, **model_opts)

    raw_opts = self.config['raw']
    weight = raw_opts['weight'][:]  # copy
    weight_t = torch.from_numpy(np.array(weight, dtype=np.float32))
    self.weight = weight
    self.aggregation = raw_opts['aggregation']
    self.aggregation_param = raw_opts['aggregation_param']
    self.aggregation_weight = Variable(weight_t, volatile=True).view(-1, 1, 1).cuda(self.config['gpu_ids'][0])

  def compute_loss(self):
    """Compute loss function. Iterate over multiple output"""
    preds = self.predictions
    weights = self.weight
    if not isinstance(preds, collections.Sequence):
      preds = [preds]
      weights = [1]

    loss = 0
    for lmda, prediction in zip(weights, preds):
      if lmda == 0:
        continue
      loss += lmda * self.criterion(prediction, self.target)

    self.loss = loss

  def aggregate_output(self):
    """Given a list of predictions from net, make a decision based on aggreagation rule"""
    if isinstance(self.predictions, collections.Sequence):
      logits = []
      for pred in self.predictions:
        logit = self.net.apply_argmax_softmax(pred).unsqueeze(0)
        logits.append(logit)

      logits = torch.cat(logits, 0)
      if self.aggregation == 'max':
        self.pred = logits.data.max(0)[0].max(1)
      elif self.aggregation == 'mean':
        self.pred = logits.data.mean(0).max(1)
      elif self.aggregation == 'weighted_mean':
        self.pred = (self.aggregation_weight.expand_as(logits) * logits).data.mean(0).max(1)
      elif self.aggregation == 'idx':
        self.pred = logits[self.aggregation_param].data.max(1)
    else:
      # Apply a softmax and return a segmentation map
      self.logits = self.net.apply_argmax_softmax(self.predictions)
      self.pred = self.logits.data.max(1)

  def forward(self, split):
    if split == 'train':
      self.predictions = self.net(Variable(self.input))
    elif split in ['validation', 'test']:
      self.predictions = self.net(Variable(self.input, volatile=True))
      self.aggregate_output()

  def backward(self):
    self.compute_loss()
    self.loss.backward()

  def validate(self):
    self.net.eval()
    self.forward(split='test')
    self.compute_loss()
    self.accumulate_results()

  def update_state(self, epoch):
    """ A function that is called at the end of every epoch. Can adjust state of the network here.
    For example, if one wants to change the loss weights for prediction during training (e.g. deep supervision), """
    late_gate = self.config['raw'].get('late_gate', None)
    if late_gate is not None:
      if epoch < late_gate:
        self.weight[0] = 0
        self.weight[1] = 0
        self.logger.info('=' * 10, 'weight={}'.format(self.weight), '=' * 10)
      if epoch == late_gate:
        self.weight = self.config['raw']['weight'][:]
        weight_t = torch.from_numpy(np.array(self.weight, dtype=np.float32))
        self.aggregation_weight = Variable(weight_t, volatile=True).view(-1, 1, 1).cuda(self.config['gpu_ids'][0])
        self.logger.info('=' * 10, 'weight={}'.format(self.weight), '=' * 10)
