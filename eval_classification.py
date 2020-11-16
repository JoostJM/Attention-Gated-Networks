import logging
import numpy as np
import json
import os

import torch
from torch.utils.data import DataLoader
import torchsample.transforms as ts
from tqdm import tqdm

from dataio.loader import get_dataset
from models import get_model
from utils.error_logger import StatLogger
from utils import configure_logging


logger = logging.getLogger()


def main(**arguments):
  global logger
  # Parse input arguments
  json_filename = arguments['config']

  # Load options
  with open(json_filename) as j_fs:
    json_opts = json.load(j_fs)

  label_dir = json_opts.get('label_dir', 'label_pred')
  if arguments.get('label_dir', None) is not None:
    label_dir = arguments['label_dir']

  model_opts = json_opts['model']

  # Ensure options specify we want to evaluate a trained model
  if model_opts['isTrain'] and model_opts['which_epoch'] == -1:
    model_opts['which_epoch'] = json_opts['training']['n_epochs']
  model_opts['isTrain'] = False
  model_opts['continue_train'] = False

  experiment = json_opts['experiment_name']

  # Set up Logging
  log_file = os.path.join(model_opts['checkpoints_dir'], experiment.replace('/', '_') + '.log')
  configure_logging(logging.INFO, slack=False, log_file=log_file)

  # Setup the NN Model
  model = get_model(experiment, **model_opts)

  errors = evaluate(model, json_opts, retry_oom=arguments['retry_oom'], test_folder=arguments['test_folder'])
  log_msg = 'Evaluation done, results:'
  for k, v in errors.items():
    if np.isscalar(v):
      log_msg += '\n\t%s: %.3f' % (k, v)
  logger.info(log_msg)


def evaluate(model, opts, retry_oom=False, test_folder='test'):
  global logger

  data_opts = opts['data']

  logger.info('Evaluating model')

  if isinstance(model, torch.nn.DataParallel):
    logger.warning('model is in DataParallel mode, but for evaluation batch size will be one. Getting root model.')
    model = model.module

  # Setup Dataset and Augmentation
  test_dataset = get_dataset([test_folder], **data_opts)[test_folder]

  test_dataset.transform = ts.Compose([ts.PadFactorSimpleITK(factor=model.config['division_factor']),
                                       ts.NormalizeSimpleITK(norm_flag=[True, False]),
                                       ts.SimpleITKtoTensor(),
                                       ts.ChannelsFirst(),
                                       ts.TypeCast(['float', 'long'])
                                       ])
  test_loader = DataLoader(dataset=test_dataset, num_workers=4, batch_size=1, shuffle=False)

  # Setup stats logger
  stat_logger = StatLogger()

  oom_iters = []

  for iteration, (image, label) in tqdm(enumerate(test_loader, 1), total=len(test_loader)):
    try:
      # Make a forward pass with the model
      if hasattr(torch, 'no_grad'):
        with torch.no_grad():
          model.set_input(image, label)
          model.validate()
      else:
        model.set_input(image, label)
        model.validate()

    except RuntimeError as e:
      if 'cuda' in str(e).lower():  # out of memory
        logger.warning('Ran out of GPU memory for case %i!', iteration)
        logger.debug('Error details', exc_info=True)
        oom_iters.append(iteration)
        torch.cuda.empty_cache()
      else:
        raise

    del image
    del label
    if model.use_cuda:
      torch.cuda.empty_cache()

  if len(oom_iters) > 0 and retry_oom:
    if isinstance(model.net, torch.nn.DataParallel):
      model.net = model.net.module
    model.net.cpu()
    model.gpu_ids = None
    model.use_cuda = False

    for iteration in tqdm(oom_iters):
      image, label = test_dataset[iteration - 1]
      image = image.expand(1, *image.size())
      label = label.expand(1, *label.size())

      # Make a forward pass with the model
      if hasattr(torch, 'no_grad'):
        with torch.no_grad():
          model.set_input(image, label)
          model.validate()
      else:
        model.set_input(image, label)
        model.validate()

      del image
      del label

  stat_logger.update(model.get_classification_stats(), 'test')

  stat_logger.statlogger2csv(split='test', out_csv_name=os.path.join(model.save_dir, 'stats.csv'))
  return stat_logger.get_errors('test')


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(description='CNN Seg Training Function')

  parser.add_argument('--config', '-c', action='append', help='training config file', required=True)
  parser.add_argument('-ro', '--retry-oom', action='store_true',
                      help='If specified, will retry evaluation on cpu '
                           'of cases that gave a cuda out-of-memory exception')
  parser.add_argument('--test-folder', '-tf', default='test')
  parser.add_argument('--label-dir', '-ld', default=None)
  
  args = parser.parse_args()

  config_dict = args.__dict__.copy()
  
  for cfg in config_dict.pop('config'):
    main(config=cfg, **config_dict)
