import torch
from torch.autograd import Variable
import torch.optim as optim

from collections import OrderedDict
import utils.util as util
from .base_model import BaseModel
from .layers.loss import *
from .utils import segmentation_stats, get_criterion
from .networks.utils import HookBasedFeatureExtractor


class FeedForwardSegmentation(BaseModel):

  def name(self):
    return 'FeedForwardSegmentation'

  def __init__(self, experiment, **model_opts):
    super(FeedForwardSegmentation, self).__init__(experiment, **model_opts)

  def get_criterion(self, **train_opts):
      return get_criterion(train_opts['criterion'], 'seg', self)

  def test(self):
    self.net.eval()
    self.forward(split='test')
    self.pred_seg = self.compute_logits().data.max(1)[1].unsqueeze(1)

  def get_segmentation_stats(self):
    self.seg_scores, self.dice_score = segmentation_stats(self.prediction, self.target)
    seg_stats = [('Overall_Acc', self.seg_scores['overall_acc']), ('Mean_IOU', self.seg_scores['mean_iou'])]
    for class_id in range(self.dice_score.size):
      seg_stats.append(('Class_{}'.format(class_id), self.dice_score[class_id]))
    return OrderedDict(seg_stats)

  def get_current_errors(self):
    if (len(self.loss.shape) == 0):
      return OrderedDict([('Seg_Loss', self.loss.item())
                          ])
    else:
      return OrderedDict([('Seg_Loss', self.loss.data[0])
                          ])

  def get_current_visuals(self):
    inp_img = util.tensor2im(self.input, 'img')
    seg_img = util.tensor2im(self.pred_seg, 'lbl')
    return OrderedDict([('out_S', seg_img), ('inp_S', inp_img)])

  def get_feature_maps(self, layer_name, upscale):
    feature_extractor = HookBasedFeatureExtractor(self.net, layer_name, upscale)
    return feature_extractor.forward(Variable(self.input))


