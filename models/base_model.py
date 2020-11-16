import logging
import os
import re
import numpy
import torch
from torch.autograd import Variable
from .networks_other import get_n_parameters, get_scheduler, benchmark_fp_bp_time
from .networks import get_network
from .utils import get_optimizer


class BaseModel():
  def __init__(self, experiment, **model_opts):
    self.input = None
    self.target = None

    self.config = {
      'isTrain': False,
      'gpu_ids': None,
      'which_epoch': int(0),
      'path_pre_trained_model': None,
      'input_nc': 1,
      'output_nc': 4,
      'feature_scale': 4,
      'tensor_dim': '2D',
      'type': 'seg',

      # Attention
      'nonlocal_mode': 'concatenation',
      'attention_dsample': (2, 2, 2),

      # Attention Classifier
      'aggregation_mode': 'concatenation'
    }

    self.config.update(model_opts)

    gpu_ids = self.config['gpu_ids']
    self.use_cuda = gpu_ids is not None and len(gpu_ids) > 0
    self.save_dir = os.path.join(model_opts['checkpoints_dir'], experiment)
    if not os.path.isdir(self.save_dir):
      os.makedirs(self.save_dir)
    elif self.config['isTrain'] and not self.config['continue_train']:
      mod_pattern = re.compile(r'(?P<epoch>\d{4})_net_\w.pth')
      epochs = [int(mod_pattern.fullmatch(m).groupdict()['epoch'])
                for m in os.listdir(self.save_dir) if mod_pattern.fullmatch(m)]
      if len(epochs) > 0:
        self.config['continue_train'] = True
        self.config['which_epoch'] = max(epochs)
    self.logger = logging.getLogger(str(self.__module__))

    self.net = get_network(**self.config)
    if self.use_cuda:
      self.net = self.net.cuda(gpu_ids[0])
      if len(gpu_ids) > 1:
        self.net = torch.nn.DataParallel(self.net, gpu_ids, gpu_ids[0])

    # load the model if a path is specified or it is in inference mode
    if not self.config['isTrain'] or self.config['continue_train']:
      self.which_epoch = self.config['which_epoch']
      if self.config['path_pre_trained_model'] is not None:
        self.load_network_from_path(self.net, self.config['path_pre_trained_model'], strict=False)
      else:
        self.load_network(self.net, 'S', self.which_epoch)
    else:
      self.which_epoch = int(0)

    self.logger.info('Network %s initialized:\n%sTotal Number of parameters %i',
                     self.config['architecture'],
                     str(self.net) + '\n' if self.config.get('print_network', True) else '',
                     get_n_parameters(self.net))

  def name(self):
    return 'BaseModel'

  def initialize_training(self, **train_opts):
    self.criterion = self.get_criterion(**train_opts)
    # initialize optimizer
    self.optimizer = get_optimizer(self.net.parameters(), **train_opts['optimizer'])
    self.scheduler = get_scheduler(self.optimizer, last_epoch=self.which_epoch, **train_opts['scheduler'])
    self.logger.info('Scheduler is added for optimizer {0}'.format(self.optimizer))

  def set_input(self, *inputs):
    # self.input.resize_(inputs[0].size()).copy_(inputs[0])
    for idx, _input in enumerate(inputs):
      # If it's a 5D array and 2D model then (B x C x H x W x Z) -> (BZ x C x H x W)
      bs = _input.size()
      if (self.config['tensor_dim'] == '2D') and (len(bs) > 4):
        _input = _input.permute(0, 4, 1, 2, 3).contiguous().view(bs[0] * bs[4], bs[1], bs[2], bs[3])

      # Define that it's a cuda array
      if idx == 0:
        self.input = _input.cuda(self.config['gpu_ids'][0]) if self.use_cuda else _input
      elif idx == 1:
        self.target = Variable(_input.cuda(self.config['gpu_ids'][0])) if self.use_cuda else Variable(_input)
        # assert self.input.shape[0] == self.target.shape[0]

  def get_criterion(self, **train_opts):
    raise NotImplementedError

  def forward(self, split):
    if split == 'train':
      self.prediction = self.net(Variable(self.input))
    elif split == 'test':
      self.prediction = self.net(Variable(self.input, volatile=True))

  def backward(self):
    self.loss = self.criterion(self.prediction, self.target)
    self.loss.backward()

  def validate(self):
    self.test()
    self.loss = self.criterion(self.prediction, self.target)

  # used in test time, no backprop
  def test(self):
    pass

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

  def compute_logits(self):
    # Apply a softmax and return the logits
    if isinstance(self.net, torch.nn.DataParallel):
      return self.net.module.apply_argmax_softmax(self.prediction)
    else:
      return self.net.apply_argmax_softmax(self.prediction)

  def get_current_visuals(self):
    return self.input

  def get_current_errors(self):
    return {}

  def save(self, epoch_label):
    self.save_network(self.net, 'S', epoch_label, self.config['gpu_ids'])

  # helper saving function that can be used by subclasses
  def save_network(self, network, network_label, epoch_label, gpu_ids):
    self.logger.info('Saving the model {0} at the end of epoch {1}'.format(network_label, epoch_label))
    if isinstance(network, torch.nn.DataParallel):
      self.logger.debug('Network in data parallel! Saving root network at network.module')
      network = network.module
    save_filename = '{0:04d}_net_{1}.pth'.format(epoch_label, network_label)
    save_path = os.path.join(self.save_dir, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if gpu_ids and len(gpu_ids) > 0 and torch.cuda.is_available():
      network.cuda(gpu_ids[0])

  # helper loading function that can be used by subclasses
  def load_network(self, network, network_label, epoch_label):
    self.logger.info('Loading the model {0} - epoch {1}'.format(network_label, epoch_label))
    if isinstance(network, torch.nn.DataParallel):
      self.logger.debug('Network in data parallel! Loading to network at network.module')
      network = network.module
    save_filename = '{0:04d}_net_{1}.pth'.format(epoch_label, network_label)
    save_path = os.path.join(self.save_dir, save_filename)
    network.load_state_dict(torch.load(save_path))

  def load_network_from_path(self, network, network_filepath, strict):
    if isinstance(network, torch.nn.DataParallel):
      self.logger.debug('Network in data parallel! Loading to network at network.module')
      network = network.module
    network_label = os.path.basename(network_filepath)
    epoch_label = network_label.split('_')[0]
    self.logger.info('Loading the model {0} - epoch {1}'.format(network_label, epoch_label))
    network.load_state_dict(torch.load(network_filepath), strict=strict)

  # update learning rate (called once every epoch)
  def update_learning_rate(self, metric=None, epoch=None):
    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
      self.scheduler.step(metrics=metric)
    else:
      self.scheduler.step()
    lr = self.optimizer.param_groups[0]['lr']
    self.logger.info('current learning rate = %.7f' % lr)

  # returns the number of trainable parameters
  def get_number_parameters(self):
    return get_n_parameters(self.net)

  # clean up the GPU memory
  def destructor(self):
    del self.net
    del self.input

  # returns the fp/bp times of the model
  def get_fp_bp_time(self, size=None):
    if size is None:
      size = (1, 1, 160, 160, 96)

    inp_array = Variable(torch.zeros(*size))
    out_array = Variable(torch.zeros(*size))
    if self.use_cuda:
      inp_array = inp_array.cuda(self.config['gpu_ids'][0])
      out_array = out_array.cuda(self.config['gpu_ids'][0])
    fp, bp = benchmark_fp_bp_time(self.net, inp_array, out_array, n_trial=50, show_pbar=True)

    bsize = size[0]
    return fp / float(bsize), bp / float(bsize)
