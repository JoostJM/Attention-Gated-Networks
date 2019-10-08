import torch
from torch.autograd import Variable
import torch.optim as optim

from collections import OrderedDict
import utils.util as util
from .base_model import BaseModel
from .layers.loss import *
from .networks_other import get_scheduler, print_network, benchmark_fp_bp_time
from .utils import segmentation_stats, get_criterion
from .networks.utils import HookBasedFeatureExtractor


class FeedForwardSegmentation(BaseModel):

  def name(self):
    return 'FeedForwardSegmentation'

  def __init__(self, experiment, **model_opts):
    super(FeedForwardSegmentation, self).__init__(experiment, **model_opts)

  def get_criterion(self, **train_opts):
      return get_criterion(train_opts['criterion'], 'seg', self)

  def set_input(self, *inputs):
    # self.input.resize_(inputs[0].size()).copy_(inputs[0])
    for idx, _input in enumerate(inputs):
      # If it's a 5D array and 2D model then (B x C x H x W x Z) -> (BZ x C x H x W)
      bs = _input.size()
      if (self.config['tensor_dim'] == '2D') and (len(bs) > 4):
        _input = _input.permute(0, 4, 1, 2, 3).contiguous().view(bs[0] * bs[4], bs[1], bs[2], bs[3])

      # Define that it's a cuda array
      if idx == 0:
        self.input = _input.cuda() if self.use_cuda else _input
      elif idx == 1:
        self.target = Variable(_input.cuda()) if self.use_cuda else Variable(_input)
        # assert self.input.size() == self.target.size()

  def forward(self, split):
    if split == 'train':
      self.prediction = self.net(Variable(self.input))
    elif split == 'test':
      self.prediction = self.net(Variable(self.input, volatile=True))
      # Apply a softmax and return a segmentation map
      if isinstance(self.net, torch.nn.DataParallel):
        self.logits = self.net.module.apply_argmax_softmax(self.prediction)
      else:
        self.logits = self.net.apply_argmax_softmax(self.prediction)
      self.pred_seg = self.logits.data.max(1)[1].unsqueeze(1)

  def backward(self):
    self.loss_S = self.criterion(self.prediction, self.target)
    self.loss_S.backward()

  def optimize_parameters(self, iteration, accumulate_iters=1):
    if iteration == 1:
      self.optimizer.zero_grad()

    self.net.train()
    self.forward(split='train')
    self.backward()

    # Check to see if the network parameters should be updated
    # If not, gradients are accumulated
    if iteration % accumulate_iters == 0:
      self.optimizer.step()
      self.optimizer.zero_grad()

  def test(self):
    self.net.eval()
    self.forward(split='test')

  def validate(self):
    self.net.eval()
    self.forward(split='test')
    self.loss_S = self.criterion(self.prediction, self.target)

  def get_segmentation_stats(self):
    self.seg_scores, self.dice_score = segmentation_stats(self.prediction, self.target)
    seg_stats = [('Overall_Acc', self.seg_scores['overall_acc']), ('Mean_IOU', self.seg_scores['mean_iou'])]
    for class_id in range(self.dice_score.size):
      seg_stats.append(('Class_{}'.format(class_id), self.dice_score[class_id]))
    return OrderedDict(seg_stats)

  def get_current_errors(self):
    if (len(self.loss_S.shape) == 0):
      return OrderedDict([('Seg_Loss', self.loss_S.item())
                          ])
    else:
      return OrderedDict([('Seg_Loss', self.loss_S.data[0])
                          ])

  def get_current_visuals(self):
    inp_img = util.tensor2im(self.input, 'img')
    seg_img = util.tensor2im(self.pred_seg, 'lbl')
    return OrderedDict([('out_S', seg_img), ('inp_S', inp_img)])

  def get_feature_maps(self, layer_name, upscale):
    feature_extractor = HookBasedFeatureExtractor(self.net, layer_name, upscale)
    return feature_extractor.forward(Variable(self.input))

  # returns the fp/bp times of the model
  def get_fp_bp_time(self, size=None):
    if size is None:
      size = (1, 1, 160, 160, 96)

    inp_array = Variable(torch.zeros(*size))
    out_array = Variable(torch.zeros(*size))
    if self.use_cuda:
      inp_array = inp_array.cuda()
      out_array = out_array.cuda()
    fp, bp = benchmark_fp_bp_time(self.net, inp_array, out_array, n_trial=50, show_pbar=True)

    bsize = size[0]
    return fp / float(bsize), bp / float(bsize)

  def save(self, epoch_label):
    self.save_network(self.net, 'S', epoch_label, self.config['gpu_ids'])
