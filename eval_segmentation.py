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


def main(arguments):
  global logger
  # Parse input arguments
  json_filename = arguments.config

  # Load options
  with open(json_filename) as j_fs:
    json_opts = json.load(j_fs)

  model_opts = json_opts['model']

  experiment = json_opts['experiment_name']

  # Set up Logging
  log_file = os.path.join(model_opts['checkpoints_dir'], experiment + '.log')
  configure_logging(logging.INFO, slack=False, log_file=log_file)

  # Setup the NN Model
  model = get_model(experiment, **model_opts)

  eval(model, json_opts, force=arguments.force)


def eval(model, opts, label_dir='label_pred', force=False):
  global logger

  data_opts = opts['data']

  logger.info('Evaluating model')

  if isinstance(model, torch.nn.DataParallel):
    logger.warning('model is in DataParallel mode, but for evaluation batch size will be one. Getting root model.')
    model = model.module

  # Setup Dataset and Augmentation
  test_dataset = get_dataset(['test'], **data_opts)['test']
  test_loader = DataLoader(dataset=test_dataset, num_workers=4, batch_size=1, shuffle=False)

  # Setup output directory
  out_dir = os.path.join(model.save_dir, label_dir)

  if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

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
        if str(e).startswith('cuda runtime error (2)'):  # out of memory
          logger.warning('Ran out of GPU memory for case %i! Will retry on cpu later...', iteration)
          oom_iters.append(iteration)
        else:
          raise

    del image
    del label
    if model.use_cuda:
      torch.cuda.empty_cache()

  if len(oom_iters) > 0:
    if isinstance(model.net, torch.nn.DataParallel):
      model.net = model.net.module
    model.net.cpu()
    model.gpu_ids = None
    model.use_cuda = False

    for iteration in oom_iters:
      image, label = test_dataset[iteration - 1]
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

  parser.add_argument('config', help='training config file')
  parser.add_argument('-f', '--force', help='If specified, overwrites existing files, otherwise, skips those cases')
  args = parser.parse_args()

  main(args)
