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
from utils import configure_logging

logger = logging.getLogger()


def main(**arguments):
  global logger
  # Parse input arguments
  json_filename = arguments['config']

  # Load options
  with open(json_filename) as j_fs:
    json_opts = json.load(j_fs)

  label_dir = json_opts.get('map_dir', 'att_maps')
  if arguments.get('map_dir', None) is not None:
    label_dir = arguments['map_dir']
    
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

  visualize(model, json_opts, map_dir=label_dir, force=arguments['force'], retry_oom=arguments['retry_oom'], test_folder=arguments['test_folder'])


def visualize(model, opts, map_dir='att_maps', layers=('attentionblock2', 'attentionblock3', 'attentionblock4'),
              force=False, retry_oom=False, test_folder='test'):
  global logger

  data_opts = opts['data']

  logger.info('Extracting Attention Maps')

  if isinstance(model, torch.nn.DataParallel):
    logger.warning('model is in DataParallel mode, but for evaluation batch size will be one. Getting root model.')
    model = model.module

  # Setup output directory
  out_dir = os.path.join(model.save_dir, map_dir)

  if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

  # Setup Dataset and Augmentation
  test_dataset = get_dataset([test_folder], **data_opts)[test_folder]
  if not force:
    for idx in range(len(test_dataset) - 1, -1, -1):
      im_path = test_dataset.image_filenames[idx]
      dest_file = os.path.join(out_dir, os.path.splitext(os.path.basename(im_path))[0]) + r'_att_map_%s.nrrd' % layers[
        0]
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

  oom_iters = []

  for iteration, (image, label) in tqdm(enumerate(test_loader, 1), total=len(test_loader)):
    try:
      process_image(out_dir, model, image, test_dataset, layers, iteration, force)
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
      process_image(out_dir, model, image, test_dataset, layers, iteration, force)

      del image
      del label


def process_image(out_dir, model, image, test_dataset, layers, iteration, force):
    im_path = test_dataset.image_filenames[iteration - 1]
    label_path = test_dataset.target_filenames[iteration - 1]
    for l in layers:
      inp_file = os.path.join(out_dir, os.path.splitext(os.path.basename(im_path))[0]) + '_inp_att_map_%s_0.nrrd' % l
      att_file = os.path.join(out_dir, os.path.splitext(os.path.basename(im_path))[0]) + '_att_map_%s_0.nrrd' % l
      inp_fmap_im, out_fmap_im = get_att_map(model, image, l, label_path)
      if force or not os.path.isfile(att_file):
        for i_idx, i in enumerate(inp_fmap_im):
          print('inp layer %s, idx %i, size %s, pixeltype %s' % (l, i_idx, i.GetSize(), i.GetPixelIDTypeAsString()))
          sitk.WriteImage(i, inp_file.replace('0.nrrd', '%i.nrrd' % i_idx), True)
        for o_idx, o in enumerate(out_fmap_im):
          print('out layer %s, idx %i, size %s, pixeltype %s' % (l, o_idx, o.GetSize(), i.GetPixelIDTypeAsString()))
          sitk.WriteImage(o, att_file.replace('0.nrrd', '%i.nrrd' % o_idx), True)


def get_att_map(model, image, layer_name, label_path):
  if hasattr(torch, "no_grad"):
    with torch.no_grad():
      model.set_input(image)
      inp_fmap, out_fmap = model.get_feature_maps(layer_name, False)
  else:
    model.set_input(image)
    inp_fmap, out_fmap = model.get_feature_maps(layer_name, False)

  division_factor = model.config['division_factor']
  label_map = sitk.ReadImage(label_path)
  label_size = np.array(label_map.GetSize())  # x, y, z
  pad = np.remainder(label_size, division_factor)

  label_l_padding = np.ceil(pad / 2.)  # x, y, z
  label_size_padded = label_size + pad

  spacing = np.array(label_map.GetSpacing())
  direction = label_map.GetDirection()

  new_origin = label_map.TransformContinuousIndexToPhysicalPoint(list(label_l_padding * -1))

  if isinstance(inp_fmap, list):
    inp_fmap_im = [sitk.GetImageFromArray(np.squeeze(i.cpu().numpy().transpose((0, 2, 3, 4, 1)))) for i in inp_fmap]
  else:
    inp_fmap_im = [sitk.GetImageFromArray(np.squeeze(inp_fmap.cpu().numpy().transpose((0, 2, 3, 4, 1))))]

  if isinstance(out_fmap, list):
    out_fmap_im = [sitk.GetImageFromArray(np.squeeze(o.cpu().numpy().transpose((0, 2, 3, 4, 1)))) for o in out_fmap]
  else:
    out_fmap_im = [sitk.GetImageFromArray(np.squeeze(out_fmap.cpu().numpy().transpose((0, 2, 3, 4, 1))))]

  for im in inp_fmap_im:
    im_size = im.GetSize()[0]  # x
    f = label_size_padded[0] / im_size

    im.SetSpacing(list(spacing * f))
    im.SetOrigin(new_origin)
    im.SetDirection(direction)

  for im in out_fmap_im:
    im_size = im.GetSize()[0]  # x
    f = label_size_padded[0] / im_size

    im.SetSpacing(list(spacing * f))
    im.SetOrigin(new_origin)
    im.SetDirection(direction)

  return inp_fmap_im, out_fmap_im


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(description='CNN Seg Training Function')

  parser.add_argument('--config', '-c', action='append', help='training config file', required=True)
  parser.add_argument('-f', '--force', action='store_true',
                      help='If specified, overwrites existing files, otherwise, skips those cases')
  parser.add_argument('-ro', '--retry-oom', action='store_true',
                      help='If specified, will retry evaluation on cpu '
                           'of cases that gave a cuda out-of-memory exception')
  parser.add_argument('--test-folder', '-tf', default='test')
  parser.add_argument('--map-dir', '-md', default=None)
  args = parser.parse_args()

  config_dict = args.__dict__.copy()
  
  for cfg in config_dict.pop('config'):
    main(config=cfg, **config_dict)
