import logging
import numpy as np
import os

import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader
import torchsample.transforms as ts
from tqdm import tqdm

from dataio.loader import get_dataset, get_dataset_path
from utils.util import json_file_to_pyobj
from models import get_model
from utils.metrics import dice_score, distance_metric, precision_and_recall
from utils.error_logger import StatLogger
from utils import _configure_logging


def eval(arguments):
  # Parse input arguments
  json_filename = arguments.config

  # Load options
  json_opts = json_file_to_pyobj(json_filename)
  train_opts = json_opts.training

  # Architecture type
  arch_type = train_opts.arch_type

  experiment = json_opts.model.experiment_name

  # Set up Logging
  log_file = os.path.join(json_opts.model.checkpoints_dir, experiment + '.log')
  _configure_logging(logging.INFO, True, log_file)
  logger = logging.getLogger()

  # Setup Dataset and Augmentation
  ds_class = get_dataset(arch_type)
  ds_path = get_dataset_path(arch_type, json_opts.data_path)
  ds_transform = ts.Compose([ts.PadFactorSimpleITK(factor=16),
                             ts.NormalizeSimpleITK(norm_flag=[True, False]),
                             ts.SimpleITKtoTensor(),
                             ts.ChannelsFirst(),
                             ts.TypeCast(['float', 'long'])
                             ])

  # Setup the NN Model
  model = get_model(json_opts.model)

  test_dataset = ds_class(ds_path, split='test', transform=ds_transform)
  test_loader = DataLoader(dataset=test_dataset, num_workers=4, batch_size=1, shuffle=False)

  # Setup stats logger
  stat_logger = StatLogger()

  for iteration, (image, label) in tqdm(enumerate(test_loader, 1), total=len(test_loader)):
    im_name = os.path.splitext(os.path.basename(test_dataset.image_filenames[iteration-1]))[0]

    with torch.no_grad():
      model.set_input(image, label)
      model.test()

    # re-load the target file to obtain geometric information
    label_path = test_dataset.target_filenames[iteration-1]
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

    label_dir = os.path.dirname(label_path)
    out_dir = label_dir + '_pred'
    if not os.path.isdir(out_dir):
      os.makedirs(out_dir)

    sitk.WriteImage(pred_seg, os.path.join(out_dir, im_name + '_label_pred.nrrd'), True)

    del image
    del label
    if model.use_cuda:
      torch.cuda.empty_cache()

  stat_logger.statlogger2csv(split='test', out_csv_name=os.path.join(model.save_dir, 'stats.csv'))
  for key, (mean_val, std_val) in stat_logger.get_errors(split='test').items():
    print('-', key, ': \t{0:.3f}+-{1:.3f}'.format(mean_val, std_val), '-')


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(description='CNN Seg Training Function')

  parser.add_argument('-c', '--config', help='training config file', required=True)
  args = parser.parse_args()

  eval(args)
