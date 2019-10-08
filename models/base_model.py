import logging
import os
import numpy
import torch
from .networks_other import get_n_parameters, get_scheduler
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

    self.use_cuda = self.config['gpu_ids'] is not None and len(self.config['gpu_ids']) > 0
    self.save_dir = os.path.join(model_opts['checkpoints_dir'], experiment)
    if not os.path.isdir(self.save_dir):
      os.makedirs(self.save_dir)
    self.logger = logging.getLogger(str(self.__module__))

    self.net = get_network(**self.config)
    if self.use_cuda:
      gpu_ids = self.config['gpu_ids']
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

    self.logger.info('Network %s initialized:\n%s\nTotal Number of parameters %i',
                     self.config['architecture'], str(self.net), get_n_parameters(self.net))

  def name(self):
    return 'BaseModel'

  def initialize_training(self, **train_opts):
    self.criterion = self.get_criterion(**train_opts)
    # initialize optimizer
    self.optimizer = get_optimizer(self.net.parameters(), **train_opts['optimizer'])
    self.scheduler = get_scheduler(self.optimizer, last_epoch=self.which_epoch, **train_opts['scheduler'])
    self.logger.info('Scheduler is added for optimiser {0}'.format(self.optimizer))

  def get_criterion(self, **train_opts):
    raise NotImplementedError

  def set_input(self, input):
    self.input = input

  def forward(self, split):
    pass

  # used in test time, no backprop
  def test(self):
    pass

  def get_image_paths(self):
    pass

  def optimize_parameters(self, iteration, accumulate_iters=1):
    pass

  def get_current_visuals(self):
    return self.input

  def get_current_errors(self):
    return {}

  def get_input_size(self):
    return self.input.size() if self.input else None

  def save(self, label):
    pass

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
    for scheduler in self.schedulers:
      if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(metrics=metric)
      else:
        scheduler.step()
      lr = self.optimizers[0].param_groups[0]['lr']
    self.logger.info('current learning rate = %.7f' % lr)

  # returns the number of trainable parameters
  def get_number_parameters(self):
    return get_n_parameters(self.net)

  # clean up the GPU memory
  def destructor(self):
    del self.net
    del self.input
