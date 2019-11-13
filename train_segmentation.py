import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import json
import os
import logging

from dataio.loader import get_dataset
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
  configure_logging(logging.INFO, slack=arguments.slack, log_file=log_file)
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
    accumulate_iter = train_opts.get("accumulate_iter", 1)
    if accumulate_iter > 1:
      logger.info('Accumulating gradients every %d iters', accumulate_iter)

    assert model.which_epoch < train_opts['n_epochs'], \
        'Model training already at designated number of epochs (%i)' % train_opts['n_epochs']
    for epoch in range(model.which_epoch + 1, train_opts['n_epochs'] + 1):
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
      if epoch % train_opts['save_epoch_freq'] == 0:
        slack_logger.info('(Experiment %s) Saving model at epoch %04d, loss:%s',
                          experiment, epoch, _get_loss_msg(error_logger))

        model.save(epoch)

      error_logger.reset()

      # Update the model learning rate
      model.update_learning_rate()

      if model.use_cuda:
        torch.cuda.empty_cache()

    # Store the final model
    slack_logger.info('(Experiment %s) Training finished! Saving model...',
                      experiment)
    model.save(epoch)

    if arguments.eval:
      import eval_segmentation
      eval_segmentation.eval(model, json_opts)
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

  parser.add_argument('-c', '--config', help='training config file', required=True)
  parser.add_argument('-d', '--debug', help='returns number of parameters and bp/fp runtime', action='store_true')
  parser.add_argument('-s', '--slack', help='enables logging to Slack Messenger', action='store_true')
  parser.add_argument('-e', '--eval', help='enables creating evaluation of the final model', action='store_true')

  args = parser.parse_args()

  train(args)
