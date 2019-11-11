import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from copy import deepcopy
from queue import Queue
import json
import os
import logging
import multiprocessing
import time
import threading

import six

from dataio.loader import get_dataset
from utils.visualiser import Visualiser
from utils.error_logger import ErrorLogger
from utils import configure_logging

from models import get_model
from hyperoptimization import HyperSpace, deep_update

_log_is_init = False


def main(arguments):
  # Parse input arguments
  json_filename = arguments.config

  # Load options
  with open(json_filename) as j_fs:
    json_opts = json.load(j_fs)

  experiment = json_opts['experiment_name']
  model_opts = json_opts['model']
  batchSize = json_opts['training'].get('batchSize', 1)
  maxBatchSize = json_opts['hyperspace']['maxBatchSize']

  json_opts['training'].update(HyperSpace.compute_batch_size(batchSize, maxBatchSize))

  out_dir = os.path.join(model_opts['checkpoints_dir'], experiment)

  # Set up Logging
  if not os.path.isdir(out_dir):
    os.makedirs(out_dir)
  log_file = os.path.join(out_dir, experiment + '.log')
  log_config, log_listener = configure_logging(logging.INFO, slack=arguments.slack, log_file=log_file, thread_safe=True)
  logger = logging.getLogger()
  slack_logger = logging.getLogger('slack')

  try:
    hyperspace = HyperSpace(json_opts['hyperspace'])
    hyperspace.save_space(os.path.join(out_dir, 'hyperspace.csv'))

    by_batch = {}
    for h_idx, h in hyperspace.hyperspace.iterrows():
      b = h.get('batchSize', batchSize)
      if b not in by_batch:
        by_batch[b] = Queue(-1)
      by_batch[b].put((h_idx, h))

    batch_sizes = sorted(by_batch.keys(), reverse=True)
    gpu_ids = model_opts.get('gpu_ids', None)
    if gpu_ids is None:
      slots_used = {None: 0}
    else:
      slots_used = {gpu: 0 for gpu in gpu_ids}

    workers = []
    while len(by_batch) > 0:
      for gpu in slots_used:
        while slots_used[gpu] < maxBatchSize:
          slot_space = maxBatchSize - slots_used[gpu]
          i = 0
          while i < len(batch_sizes) and batch_sizes[i] > slot_space:
            i += 1

          if i == len(batch_sizes):
            # No batch sizes left that would fit in the remaining space...
            break

          worker_batchSize = batch_sizes[i]
          parent_results_pipe, child_results_pipe = multiprocessing.Pipe()

          config_idx, config = by_batch[worker_batchSize].get()
          if by_batch[worker_batchSize].empty():
            del by_batch[worker_batchSize]
            del batch_sizes[i]

          slack_logger.info('Starting experiment for config %i (worker %i))',
                      config_idx, len(workers))

          config_opts = deepcopy(json_opts)

          config_opts['experiment_name'] = experiment + '/' + str(config_idx)

          config_opts['training'] = deep_update(config_opts['training'], config)
          config_opts['log_config'] = log_config
          config_opts['results'] = child_results_pipe
          if gpu is not None:
            config_opts['model']['gpu_ids'] = [gpu]

          p = multiprocessing.Process(target=train, args=(config_opts,))
          p.start()
          workers.append((gpu, worker_batchSize, p, parent_results_pipe))
          slots_used[gpu] += worker_batchSize

      while np.all([p[2].is_alive() for p in workers]):
        time.sleep(30)

      for i in range(len(workers) - 1, -1, -1):
        if workers[i][2].is_alive():
          continue
        # worker is done! clear up the space for new worker
        logger.info('Worker %i is done!', i)
        gpu, worker_batchSize, p, results_pipe = workers[i]
        p.join()
        slots_used[gpu] -= worker_batchSize

        # Store the results in the hyperspace
        if results_pipe.poll(3):
          hyperspace.add_result(results_pipe.recv())
          hyperspace.save_space(os.path.join(out_dir, 'hyperspace.csv'))

        # Remove the worker from the list
        results_pipe.close()
        del p
        del results_pipe
        del workers[i]

  except Exception:
    slack_logger.critical('(Experiment %s) Oh No! Hyperspace enumeration failed!!', experiment, exc_info=True)
  finally:
    if log_listener is not None:
      log_listener.stop()


def train(json_opts):
  global _log_is_init
  log_config = json_opts['log_config']
  if not _log_is_init:
    from logging import config
    config.dictConfig(log_config)

  logger = logging.getLogger()
  slack_logger = logging.getLogger('slack')

  experiment = json_opts['experiment_name']
  config_idx = int(experiment.split('/')[-1])
  data_opts = json_opts['data']
  train_opts = json_opts['training']
  model_opts = json_opts['model']

  batchSize = train_opts.get('batchSize', 1)
  results_pipe = json_opts['results']
  results = {config_idx: {}}

  # Set the current thread name to the name of the experiment
  threading.current_thread().setName(experiment)

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

  # Setup Data and Augmentation
  datasets = get_dataset(['train', 'validation'], **data_opts)
  train_loader = DataLoader(dataset=datasets['train'], num_workers=4, batch_size=batchSize, shuffle=True)
  valid_loader = DataLoader(dataset=datasets['validation'], num_workers=4, batch_size=batchSize, shuffle=False)

  # Visualisation Parameters
  visualizer = Visualiser(json_opts['visualisation'], save_dir=model.save_dir)
  error_logger = ErrorLogger()

  # Training Function
  slack_logger.info('Starting training for experiment %s config:\n%s', experiment, dict_pretty_print(train_opts))
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
        for epoch_iter, (images, labels) in enumerate(train_loader, 1):
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
        for epoch_iter, (images, labels) in enumerate(valid_loader, 1):
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

        # Update the plots and results_dict
        for split in error_logger.variables.keys():
          visualizer.plot_current_errors(epoch, error_logger.get_errors(split), split_name=split)
          visualizer.print_current_errors(epoch, error_logger.get_errors(split), split_name=split)
          results[config_idx][split] = error_logger.get_errors(split)

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
      slack_logger.info('(Experiment %s) Pre-training finished!', experiment)
      if epoch % train_opts['save_epoch_freq'] != 0:  # Only save when not done so already
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
          results[config_idx][split] = error_logger.get_errors(split)

        error_logger.reset()

      swa_scheduler = get_swa_scheduler(model.optimizer, epoch, **train_opts)

      for epoch in range(epoch + 1, train_opts['n_epochs'] + 1):
        logger.info('(epoch: %d, total # iters: %d)' % (epoch, len(train_loader)))

        # Set Learning rate
        swa_scheduler.step()
        logger.info('SWA learning rate: %.7f', model.optimizer.param_groups[0]['lr'])

        # Training Iterations
        for epoch_iter, (images, labels) in enumerate(train_loader, 1):
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
        for epoch_iter, (images, labels) in enumerate(valid_loader, 1):
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
          results[config_idx][split] = error_logger.get_errors(split)

        # Save the model parameters
        if epoch % train_opts['save_epoch_freq'] == 0:
          slack_logger.info('(Experiment %s) Saving model at epoch %04d, loss:%s',
                            experiment, epoch, _get_loss_msg(error_logger))

          model.save(epoch)

        error_logger.reset()

      slack_logger.info('SWA Training finished!')
      if epoch % train_opts['save_epoch_freq'] != 0:  # Only save when not done so already
        model.save(epoch)
  except Exception:
    slack_logger.critical('(Experiment %s) Oh No! Training failed!!', experiment, exc_info=True)

  results_pipe.send(results)


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


def dict_pretty_print(d):
  def enum_d(d, prefix='\t'):
    out_str = ''
    for k, v in six.iteritems(d):
      if isinstance(v, dict):
        out_str += '\n%s%s:' % (prefix, k) + enum_d(v, prefix+'\t')
      else:
        out_str += '\n%s%s: %s' % (prefix, k, v)
    return out_str

  return enum_d(d)[1:]  # skip first newline


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(description='CNN Seg hyperspace optimisation Function')

  parser.add_argument('-c', '--config', help='training config file', required=True)
  parser.add_argument('-d', '--debug', help='returns number of parameters and bp/fp runtime', action='store_true')
  parser.add_argument('-s', '--slack', help='enables logging to Slack Messenger', action='store_true')

  args = parser.parse_args()

  main(args)
