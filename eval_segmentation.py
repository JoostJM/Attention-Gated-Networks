import logging
import numpy as np
import json
import os

import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader
import torchsample.transforms as ts
from tqdm import tqdm

from dataio.loader import get_dataset
from models import get_model
from utils.metrics import dice_score, distance_metric, precision_and_recall
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

  evaluate(model, json_opts, label_dir=label_dir, force=arguments['force'], retry_oom=arguments['retry_oom'], test_folder=arguments['test_folder'])


def evaluate(model, opts, label_dir='label_pred', force=False, retry_oom=False, test_folder='test'):
  global logger

  data_opts = opts['data']

  logger.info('Evaluating model')

  if isinstance(model, torch.nn.DataParallel):
    logger.warning('model is in DataParallel mode, but for evaluation batch size will be one. Getting root model.')
    model = model.module

  # Setup output directory
  out_dir = os.path.join(model.save_dir, label_dir)

  if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

  # Setup Dataset and Augmentation
  test_dataset = get_dataset([test_folder], **data_opts)[test_folder]
  if not force:
    for idx in range(len(test_dataset) - 1, -1, -1):
      im_path = test_dataset.image_filenames[idx]
      dest_file = os.path.join(out_dir, os.path.splitext(os.path.basename(im_path))[0]) + '_label_pred.nrrd'
      if os.path.isfile(dest_file):
        logger.info('File %s already exists. Skipping this case', dest_file)
        del test_dataset.image_filenames[idx]
        del test_dataset.target_filenames[idx]

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
    im_path = test_dataset.image_filenames[iteration - 1]
    label_path = test_dataset.target_filenames[iteration - 1]

    dest_file = os.path.join(out_dir, os.path.splitext(os.path.basename(im_path))[0]) + '_label_pred.nrrd'
    if force or not os.path.isfile(dest_file):
      try:
        pred_seg = make_segmentation(model, image, label, stat_logger, im_path, label_path)

        sitk.WriteImage(pred_seg, dest_file, True)
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
      im_path = test_dataset.image_filenames[iteration - 1]
      label_path = test_dataset.target_filenames[iteration - 1]
      dest_file = os.path.join(out_dir, os.path.splitext(os.path.basename(im_path))[0] + '_label_pred.nrrd')
      pred_seg = make_segmentation(model, image, label, stat_logger, im_path, label_path)

      sitk.WriteImage(pred_seg, dest_file, True)
      del image
      del label

  stat_logger.statlogger2csv(split='test', out_csv_name=os.path.join(model.save_dir, 'stats.csv'))
  for key, (mean_val, std_val) in stat_logger.get_errors(split='test').items():
    print('-', key, ': \t{0:.3f}+-{1:.3f}'.format(mean_val, std_val), '-')


def make_segmentation(model, image, label, stat_logger, image_path, label_path):
  im_name = os.path.splitext(os.path.basename(image_path)[0])

  if hasattr(torch, "no_grad"):
    with torch.no_grad():
      model.set_input(image, label)
      model.test()
  else:
    model.set_input(image, label)
    model.test()

  # re-load the target file to obtain geometric information
  label_map = sitk.ReadImage(label_path)

  pred_seg_arr = model.pred_seg.cpu().numpy()
  pred_seg_arr = np.squeeze(pred_seg_arr)
  pred_seg = sitk.GetImageFromArray(pred_seg_arr)

  # Revert any padding done during data augmentation
  crop = ts.SpecialCropSimpleITK(label_map.GetSize(), crop_type=0)
  pred_seg = crop(pred_seg)
  pred_seg.CopyInformation(label_map)

  pred_seg_arr = sitk.GetArrayFromImage(pred_seg)
  gt_arr = sitk.GetArrayFromImage(label_map)

  dice_vals = dice_score(gt_arr, pred_seg_arr, n_class=2)
  md, hd = distance_metric(gt_arr, pred_seg_arr, dx=2.00, k=2)
  precision, recall = precision_and_recall(gt_arr, pred_seg_arr, n_class=2)
  stat_logger.update(split='test', input_dict={'img_name': im_name,
                                               'dice': dice_vals[1],
                                               'precision': precision[1],
                                               'recall': recall[1],
                                               'md': md,
                                               'hd': hd
                                               })

  return pred_seg


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(description='CNN Seg Training Function')

  parser.add_argument('--config', '-c', action='append', help='training config file', required=True)
  parser.add_argument('-f', '--force', action='store_true', help='If specified, overwrites existing files, otherwise, skips those cases')
  parser.add_argument('-ro', '--retry-oom', action='store_true',
                      help='If specified, will retry evaluation on cpu '
                           'of cases that gave a cuda out-of-memory exception')
  parser.add_argument('--test-folder', '-tf', default='test')
  parser.add_argument('--label-dir', '-ld', default=None)
  
  args = parser.parse_args()

  config_dict = args.__dict__.copy()
  
  for cfg in config_dict.pop('config'):
    main(config=cfg, **config_dict)

