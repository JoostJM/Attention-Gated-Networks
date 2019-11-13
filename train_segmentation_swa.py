import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import json
import os
import logging

from dataio.loader import get_dataset
from utils.util import json_file_to_pyobj
from utils.visualiser import Visualiser
from utils.error_logger import ErrorLogger
from utils import configure_logging

from models import get_model


def train(arguments):
  # Parse input arguments
  json_filename = arguments.config
  network_debug = arguments.debug

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
  log_file = os.path.join(checkpoints_dir, experiment + '.log')
  configure_logging(logging.INFO, arguments.slack, log_file)
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
  swa_model = None
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
  datasets = get_dataset(['train', 'validation'], **data_opts)
  train_loader = DataLoader(dataset=datasets['train'], num_workers=4, batch_size=batchSize, shuffle=True)
  valid_loader = DataLoader(dataset=datasets['validation'], num_workers=4, batch_size=batchSize, shuffle=False)

  # Visualisation Parameters
  visualizer = Visualiser(experiment, json_opts['visualisation'], save_dir=model.save_dir)
  error_logger = ErrorLogger()

  # Training Function
  slack_logger.info('Starting training for experiment %s', experiment)
  try:
    model.initialize_training(**train_opts)
    accumulate_iter = getattr(train_opts, "accumulate_iter", 1)
    if accumulate_iter > 1:
      logger.info('Accumulating gradients every %d iters', accumulate_iter)

    init_n_epoch = train_opts['n_epochs']
    if train_opts.get('swa', False):
      init_n_epoch = train_opts['swa_start'] - 1

    epoch = model.which_epoch
    if model.which_epoch < init_n_epoch:
      logger.info('Pretraining the model for %i epochs', init_n_epoch - model.which_epoch)
      for epoch in range(model.which_epoch + 1, init_n_epoch + 1):
        logger.info('(epoch: %d, total # iters: %d)' % (epoch, len(train_loader)))

        # Training Iterations
        for epoch_iter, (images, labels) in tqdm(enumerate(train_loader, 1), total=len(train_loader)):
          # Make a training update
          model.set_input(images, labels)
          model.optimize_parameters(epoch_iter, accumulate_iter)

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
        for epoch_iter, (images, labels) in tqdm(enumerate(valid_loader, 1), total=len(valid_loader)):
          # Make a forward pass with the model
          if hasattr(torch, 'no_grad'):
            with torch.no_grad():
              model.set_input(images, labels)
              model.validate()
          else:
            model.set_input(images, labels)
            model.validate()

          # Error visualisation
          errors = model.get_current_errors()
          stats = model.get_segmentation_stats()
          error_logger.update({**errors, **stats}, split='validation')

          # Visualise predictions
          visuals = model.get_current_visuals()
          visualizer.display_current_results(visuals, epoch=epoch, save_result=False)
          del images
          del labels

        # Update the plots
        for split in error_logger.variables.keys():
          visualizer.plot_current_errors(epoch, error_logger.get_errors(split), split_name=split)
          visualizer.print_current_errors(epoch, error_logger.get_errors(split), split_name=split)

        # Save the model parameters
        if epoch % train_opts.save_epoch_freq == 0:
          slack_logger.info('(Experiment %s) Saving model at epoch %04d, loss:%s',
                            experiment, epoch, _get_loss_msg(error_logger))

          model.save(epoch)

        error_logger.reset()

        # Update the model learning rate
        model.update_learning_rate()

        if model.use_cuda:
          torch.cuda.empty_cache()

      # Store the final model
      slack_logger.info('(Experiment %s) Pre-training finished!', experiment)
      if epoch % train_opts.save_epoch_freq != 0:  # Only save when not done so already
        model.save(epoch)

    if train_opts.get('swa', False):
      slack_logger.info('Starting Stochastic Weight Averaging training at epoch %i', epoch)

      swa_model = get_model(experiment, **model_opts)

      # Indicate that the weights don't need gradients (weights are not optimized by backprop)
      for param in swa_model.net.parameters():
        param.requires_grad = False

      if epoch + 1 > train_opts['swa_start']:
        # Continue training, already started swa
        swa_n = (epoch - train_opts['swa_start'] + 1) // train_opts['swa_c'] + 1
        logger.info('loading SWA model at checkpoint %d', swa_n)
        swa_model.load_network(swa_model.net, 'S_SWA', swa_n)
      else:
        swa_n = 1
        update_swa(model, swa_model, swa_n, epoch, valid_loader, error_logger, visualizer)

        # Update the plots
        for split in error_logger.variables.keys():
          visualizer.plot_current_errors(epoch, error_logger.get_errors(split), split_name=split)
          visualizer.print_current_errors(epoch, error_logger.get_errors(split), split_name=split)

        error_logger.reset()

      swa_scheduler = get_swa_scheduler(model.optimizer, epoch, **train_opts)

      for epoch in range(epoch + 1, train_opts['n_epochs'] + 1):
        logger.info('(epoch: %d, total # iters: %d)' % (epoch, len(train_loader)))

        # Set Learning rate
        swa_scheduler.step()
        logger.info('SWA learning rate: %.7f', model.optimizer.param_groups[0]['lr'])

        # Training Iterations
        for epoch_iter, (images, labels) in tqdm(enumerate(train_loader, 1), total=len(train_loader)):
          # Make a training update
          model.set_input(images, labels)
          model.optimize_parameters(epoch_iter, accumulate_iter)

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
        for epoch_iter, (images, labels) in tqdm(enumerate(valid_loader, 1), total=len(valid_loader)):
          # Make a forward pass with the model
          if hasattr(torch, 'no_grad'):
            with torch.no_grad():
              model.set_input(images, labels)
              model.validate()
          else:
            model.set_input(images, labels)
            model.validate()

          # Error visualisation
          errors = model.get_current_errors()
          stats = model.get_segmentation_stats()
          error_logger.update({**errors, **stats}, split='validation')

          # Visualise predictions
          visuals = model.get_current_visuals()
          visualizer.display_current_results(visuals, epoch=epoch, save_result=False)
          del images
          del labels

        if (epoch - train_opts['swa_start'] + 1) % train_opts['swa_c'] == 0:
          swa_n += 1
          update_swa(model, swa_model, swa_n, epoch, valid_loader, error_logger, visualizer)

        # Update the plots
        for split in error_logger.variables.keys():
          visualizer.plot_current_errors(epoch, error_logger.get_errors(split), split_name=split)
          visualizer.print_current_errors(epoch, error_logger.get_errors(split), split_name=split)

        # Save the model parameters
        if epoch % train_opts['save_epoch_freq'] == 0:
          slack_logger.info('(Experiment %s) Saving model at epoch %04d, loss:%s',
                            experiment, epoch, _get_loss_msg(error_logger))

          model.save(epoch)

        error_logger.reset()

      slack_logger.info('SWA Training finished!')
      if epoch % train_opts['save_epoch_freq'] != 0:  # Only save when not done so already
        model.save(epoch)

    if arguments.eval:
      import eval_segmentation
      eval_segmentation.eval(model, json_opts)
      if train_opts.get('swa', False):
        eval_segmentation.eval(swa_model, json_opts, 'label_pred_swa')
  except Exception:
    slack_logger.critical('(Experiment %s) Oh No! Training failed!!', experiment, exc_info=True)


def update_swa(model, swa_model, swa_n, epoch, valid_loader, error_logger, visualizer):
  update_swa_weigths(model.net, swa_model.net, 1.0 / swa_n)

  # SWA Validation Iterations
  for epoch_iter, (images, labels) in tqdm(enumerate(valid_loader, 1), total=len(valid_loader)):
    # Make a forward pass with the model
    if hasattr(torch, 'no_grad'):
      with torch.no_grad():
        swa_model.set_input(images, labels)
        swa_model.validate()
    else:
      swa_model.set_input(images, labels)
      swa_model.validate()

    # Error visualisation
    errors = swa_model.get_current_errors()
    stats = swa_model.get_segmentation_stats()
    error_logger.update({**errors, **stats}, split='swa_validation')

    # Visualise predictions
    visuals = swa_model.get_current_visuals()
    visualizer.display_current_results(visuals, epoch=epoch, save_result=False)
    del images
    del labels

  # Save the SWA network
  swa_model.save_network(swa_model.net, 'S_SWA', swa_n, swa_model.gpu_ids)


def get_swa_scheduler(optimizer, last_epoch, **train_opts):
  a1 = train_opts['swa_a1']
  a2 = train_opts['swa_a2']
  ratio = a2 / a1
  c = train_opts['swa_c']
  last_epoch = last_epoch - train_opts['swa_start']

  def swa_rule(epoch):
    t = (epoch % c) / (c - 1)
    alpha = (1 - t) + t * ratio
    return alpha

  if last_epoch < 0:
    for group in optimizer.param_groups:
      group['lr'] = a1
  else:
    for group in optimizer.param_groups:
      group.setdefault('initial_lr', a1)

  scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=swa_rule, last_epoch=last_epoch)
  return scheduler


def update_swa_weigths(net, swa_net, alpha=1):
  if isinstance(net, torch.nn.DataParallel):
    net = net.module
  if isinstance(swa_net, torch.nn.DataParallel):
    swa_net = swa_net.module

  for param, swa_param in zip(net.parameters(), swa_net.parameters()):
    swa_param.data *= (1 - alpha)
    swa_param.data += param.data * alpha


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

  parser.add_argument('-c', '--config', help='training config file', required=True)
  parser.add_argument('-d', '--debug', help='returns number of parameters and bp/fp runtime', action='store_true')
  parser.add_argument('-s', '--slack', help='enables logging to Slack Messenger', action='store_true')
  parser.add_argument('-e', '--eval', help='enables creating evaluation of the final model', action='store_true')

  args = parser.parse_args()

  train(args)
