import torch.utils.data as data
import numpy as np
import datetime

import logging
from os import listdir
from os.path import isfile, join
from .utils import is_image_file

import SimpleITK as sitk
import six


class RectumSegmentationDataset(data.Dataset):
  def __init__(self, root_dir, split, transform=None, preload_data=False, **kwargs):
    super(RectumSegmentationDataset, self).__init__()

    self.logger = logging.getLogger('RectumSegmentationDataset')

    image_dir = join(root_dir, split, 'image')
    target_dir = join(root_dir, split, 'label')
    self.image_filenames = sorted([join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)])
    self.target_filenames = sorted([join(target_dir, x) for x in listdir(target_dir) if is_image_file(x)])
    assert len(self.image_filenames) == len(self.target_filenames)

    # Optional exclusion of cases (e.g. exclude low-quality scans)
    exclusion = kwargs.get('exclusion', None)
    self.apply_filter(root_dir, split, exclusion)

    # Optional channel select
    self.channels = kwargs.get('channels', None)
    if self.channels is not None:
      self.logger.info('Using channel selection %s', self.channels)
    elif isinstance(self.channels, (list, tuple)) and len(self.channels) == 1:
      self.channels = self.channels[0]

    # report the number of images in the dataset
    self.logger.info('Number of {0} images: {1}'.format(split, self.__len__()))

    # data augmentation
    self.transform = transform

    # data load into the ram memory
    self.preload_data = preload_data
    if self.preload_data:
      self.logger.info('Preloading the {0} dataset ...'.format(split))
      self.raw_images = [self._load_image(ii) for ii in self.image_filenames]
      self.raw_labels = [sitk.ReadImage(ii) for ii in self.target_filenames]
      self.logger.info('Loading is done\n')

  def apply_filter(self, root_dir, split, exclusion):
    if exclusion is None:
      return
    exclude_file = exclusion.get(split, None)
    if exclude_file is None:
      return
    exclude_file = join(root_dir, exclude_file)
    if not isfile(exclude_file):
      self.logger.warning('Exclusion file %s for split %s not found!', exclude_file, split)
      return

    exluded = set()
    with open(join(root_dir, exclude_file)) as exclude_fs:
      for r in exclude_fs.readlines():
        r = r.replace('\n', '').replace('\r', '')
        if r == '':
          continue
        exluded.add(r.lower())
      self.logger.info('Exclusion filter enabled! Read %i filenames in %s', len(exluded), exclude_file)

    if len(exluded) > 0:
      exclude_cnt = 0
      for i in range(len(self.image_filenames) - 1, -1, -1):
        im_name = self.image_filenames[i]
        for e in exluded:
          if e in im_name.lower():
            del self.image_filenames[i]
            del self.target_filenames[i]
            exclude_cnt += 1
      self.logger.info('Excluded %i cases from split %s', exclude_cnt, split)

  def __getitem__(self, index):
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    # update the seed to avoid workers sample the same augmentation parameters
    np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

    # load the images
    if not self.preload_data:
      input = self._load_image(self.image_filenames[index])
      target = sitk.ReadImage(self.target_filenames[index])
    else:
      input = self.raw_images[index]
      target = self.raw_labels[index]

    # handle exceptions
    if self.transform:
      input, target = self.transform(input, target)

    return input, target

  def __len__(self):
    return len(self.image_filenames)

  def _load_image(self, fname):
    im = sitk.ReadImage(fname)
    if self.channels is None:
      return im

    n_channels = im.GetNumberOfComponentsPerPixel()

    if isinstance(self.channels, six.integer_types):
      assert self.channels < n_channels, \
          'Only % channels available, trying to select channel at index %i' % (n_channels, self.channels)
      return sitk.VectorIndexSelectionCast(im, self.channels, sitk.sitkFloat32)
    elif isinstance(self.channels, (list, tuple)):
      assert np.max(self.channels) < n_channels, \
          'Only % channels available, trying to select channel at index %i' % (n_channels, np.max(self.channels))
      return sitk.Compose([sitk.VectorIndexSelectionCast(im, i, sitk.sitkFloat32) for i in self.channels])
