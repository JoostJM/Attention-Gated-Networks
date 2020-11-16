import json
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, sampler
from tqdm import tqdm


from dataio.loader import get_dataset
from utils.visualiser import Visualiser
from utils.error_logger import ErrorLogger
from models.networks_other import adjust_learning_rate
from utils import configure_logging

from models import get_model


class StratifiedSampler(object):
  """Stratified Sampling
  Provides equal representation of target classes in each batch
  """
  def __init__(self, class_vector, batch_size):
    """
    Arguments
    ---------
    class_vector : torch tensor
    a vector of class labels
    batch_size : integer
    batch_size
    """
    self.class_vector = class_vector
    self.batch_size = batch_size
    self.num_iter = len(class_vector) // 52
    self.n_class = 14
    self.sample_n = 2
    # create pool of each vectors
    indices = {}
    for i in range(self.n_class):
      indices[i] = np.where(self.class_vector == i)[0]

    self.indices = indices
    self.background_index = np.argmax([len(indices[i]) for i in range(self.n_class)])

  def gen_sample_array(self):
    # sample 2 from each class
    sample_array = []
    for i in range(self.num_iter):
      arrs = []
      for j in range(self.n_class):
        n = self.sample_n
        if j == self.background_index:
            n = self.sample_n * (self.n_class-1)
        arr = np.random.choice(self.indices[j], n)
        arrs.append(arr)

      sample_array.append(np.hstack(arrs))
    return np.hstack(sample_array)

  def __iter__(self):
    return iter(self.gen_sample_array())

  def __len__(self):
    return len(self.class_vector)


# Not using anymore
def check_warm_start(epoch, model, train_opts):
  if hasattr(train_opts, "warm_start_epoch"):
    if epoch < train_opts.warm_start_epoch:
      print('... warm_start: lr={}'.format(train_opts.warm_start_lr))
      adjust_learning_rate(model.optimizers[0], train_opts.warm_start_lr)
  elif epoch == train_opts.warm_start_epoch:
    print('... warm_start ended: lr={}'.format(model.opts.lr_rate))
    adjust_learning_rate(model.optimizers[0], model.opts.lr_rate)


def train(**arguments):
  # Parse input arguments
  json_filename = arguments['config']
  network_debug = arguments['debug']

  # Load options
  with open(json_filename) as j_fs:
    json_opts = json.load(j_fs)

  experiment = json_opts['experiment_name']
  data_opts = json_opts['data']
  train_opts = json_opts['training']
  model_opts = json_opts['model']

  batchSize = train_opts.get('batchSize', 1)
  checkpoints_dir = model_opts['checkpoints_dir']

  # Set up Logging
  if not os.path.isdir(checkpoints_dir):
    os.makedirs(checkpoints_dir)
  log_file = os.path.join(checkpoints_dir, experiment.replace('/', '_') + '.log')
  configure_logging(logging.INFO, slack=arguments['slack'], log_file=log_file)
  logger = logging.getLogger()
  slack_logger = logging.getLogger('slack')

  # Try to enable cudnn benchmark if cuda will be used
  try:
    if model_opts.get('gpu_ids', None) is not None and len(model_opts['gpu_ids']) > 0:
      torch.backends.cudnn.enabled = True
      torch.backends.cudnn.benchmark = True
      logger.info('CuDNN benchmark enabled')
  except Exception:
    logger.warning('Failed to enable CuDNN benchmark', exc_info=True)

  # Setup the NN Model
  model = get_model(experiment, **model_opts)
  if network_debug:
    augmentation_opts = json_opts['data']['augmentation']
    input_size = (batchSize, model_opts['input_nc'], *augmentation_opts.patch_size)
    logger.info('# of pars: %s', model.get_number_parameters())
    logger.info(
        'fp time: {0:.3f} sec\tbp time: {1:.3f} sec per sample'.format(*model.get_fp_bp_time(size=input_size)))
    logger.info('Max_memory used: {0:.3f}'.format(torch.cuda.max_memory_allocated()))
    exit()

  # Setup Data and Augmentation
  num_workers = train_opts.get('num_workers', 16)
  datasets = get_dataset(['train', 'validation', 'test'], **data_opts)

  # create sampler
  if train_opts['sampler'] == 'stratified':
    logger.info('stratified sampler')
    train_sampler = StratifiedSampler(datasets['train'].labels, batchSize)
    batch_size = 52
  elif train_opts['sampler'] == 'weighted2':
    logger.info('weighted sampler with background weight={}x'.format(train_opts['bgd_weight_multiplier']))
    # modify and increase background weight
    weight = datasets['train'].weight
    bgd_weight = np.min(weight)
    weight[abs(weight - bgd_weight) < 1e-8] = bgd_weight * train_opts['bgd_weight_multiplier']
    train_sampler = sampler.WeightedRandomSampler(weight, len(datasets['train'].weight))
    batch_size = batchSize
  else:
    logger.info('weighted sampler')
    train_sampler = sampler.WeightedRandomSampler(datasets['train'].weight, len(datasets['train'].weight))
    batch_size = batchSize

  train_loader = DataLoader(dataset=datasets['train'], num_workers=num_workers,
                            batch_size=batch_size, sampler=train_sampler)
  valid_loader = DataLoader(dataset=datasets['validation'], num_workers=num_workers,
                            batch_size=batchSize, shuffle=True)

  # Visualisation Parameters
  visualizer = Visualiser(experiment, json_opts['visualisation'], save_dir=model.save_dir)
  error_logger = ErrorLogger()

  # Training Function
  slack_logger.info('Starting training for experiment %s', experiment)
  try:
    track_labels = np.arange(len(datasets['train'].label_names))
    model.initialize_training(**train_opts)
    model.set_labels(track_labels)

    accumulate_iter = train_opts.get("accumulate_iter", 1)

    if hasattr(model, 'update_state'):
      model.update_state(0)

    for epoch in range(model.which_epoch, train_opts['n_epochs']):
      logger.info('(epoch: %d, total # iters: %d)' % (epoch, len(train_loader)))

      # # # --- Start ---
      # import matplotlib.pyplot as plt
      # plt.ion()
      # plt.figure()
      # target_arr = np.zeros(14)
      # # # --- End ---

      # Training Iterations
      for epoch_iter, (images, labels) in tqdm(enumerate(train_loader, 1), total=len(train_loader)):
        # Make a training update
        model.set_input(images, labels)
        model.optimize_parameters(epoch_iter)

        # if epoch == (train_opts['n_epochs']-1):
        #    import time
        #    time.sleep(36000)

        if train_opts['max_it'] == epoch_iter:
          break

        # # # --- visualise distribution ---
        # for lab in labels.numpy():
        #     target_arr[lab] += 1
        # plt.clf(); plt.bar(train_dataset.label_names, target_arr); plt.pause(0.01)
        # # # --- End ---

        # Visualise predictions
        if epoch_iter <= 100:
          visuals = model.get_current_visuals()
          visualizer.display_current_results(visuals, epoch=epoch, save_result=False)

        # Error visualisation
        errors = model.get_current_errors()
        error_logger.update(errors, split='train')
        del images
        del labels

        # Update the network parameters if some have been accumulated (epoch_iter % accumulate_iter != 0)
        # Reflects update from different-sized final batch
        if epoch_iter % accumulate_iter != 0:
          model.optimizer.step()

      # Validation Iterations
      model.reset_results()
      for epoch_iter, (images, labels) in tqdm(enumerate(valid_loader, 1), total=len(valid_loader)):
        # Make a forward pass with the model
        if hasattr(torch, 'no_grad'):
          with torch.no_grad():
            model.set_input(images, labels)
            model.validate()
        else:
          model.set_input(images, labels)
          model.validate()

        # Visualise predictions
        visuals = model.get_current_visuals()
        visualizer.display_current_results(visuals, epoch=epoch, save_result=False)

        if train_opts['max_it'] == epoch_iter:
          break

        # Error visualisation
        errors = model.get_accumulated_errors()
        stats = model.get_classification_stats()
        error_logger.update({**errors, **stats}, split='validation')

        del images
        del labels

      valid_err = errors['CE']

      # Update the plots
      for split in error_logger.variables.keys():
        # exclude background
        # track_labels = np.delete(track_labels, 3)
        # show_labels = train_dataset.label_names[:3] + train_dataset.label_names[4:]
        show_labels = datasets['train'].label_names
        visualizer.plot_current_errors(epoch, error_logger.get_errors(split), split_name=split, labels=show_labels)
        visualizer.print_current_errors(epoch, error_logger.get_errors(split), split_name=split)
        error_logger.reset()

      # Save the model parameters
      if epoch % train_opts['save_epoch_freq'] == 0:
        slack_logger.info('(Experiment %s) Saving model at epoch %04d, loss:%s',
                          experiment, epoch, _get_loss_msg(error_logger))
        model.save(epoch)

      if hasattr(model, 'update_state'):
        model.update_state(epoch)

      # Update the model learning rate
      model.update_learning_rate(metric=valid_err, epoch=epoch)

      if model.use_cuda:
        torch.cuda.empty_cache()

    # Store the final model
    slack_logger.info('(Experiment %s) Training finished! Saving model...',
                      experiment)
    model.save(epoch)

    if arguments['eval']:
      import eval_classification
      errors = eval_classification.evaluate(model, json_opts)
      log_msg = 'Evaluation done, results:'
      for k, v in errors.items():
        if np.isscalar(v):
          log_msg += '\n\t%s: %.3f' % (k, v)
      slack_logger.info(log_msg)
  except Exception:
    slack_logger.critical('(Experiment %s) Oh No! Training failed!!', experiment, exc_info=True)


def _get_loss_msg(error_logger):
  loss_msg = ''
  for split in error_logger.variables.keys():
    loss_msg += '\n\t (split %s)' % split
    for k, v in error_logger.get_errors(split).items():
      if np.isscalar(v):
        loss_msg += '%s: %.3f ' % (k, v)
  return loss_msg


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(description='CNN Seg Training Function')

  parser.add_argument('-c', '--config', action='append', help='training config file', required=True)
  parser.add_argument('-d', '--debug', help='returns number of parameters and bp/fp runtime', action='store_true')
  parser.add_argument('-s', '--slack', help='enables logging to Slack Messenger', action='store_true')
  parser.add_argument('-e', '--eval', help='enables creating evaluation of the final model', action='store_true')

  args = parser.parse_args()

  config_dict = args.__dict__.copy()

  for cfg in config_dict.pop('config'):
    train(config=cfg, **config_dict)
