import torch.utils.data as data
import numpy as np
import datetime

import logging
from os import listdir
from os.path import isfile, join
from .utils import is_image_file

import SimpleITK as sitk


class RectumSegmentationDataset(data.Dataset):
    def __init__(self, root_dir, split, transform=None, preload_data=False):
        super(RectumSegmentationDataset, self).__init__()

        logger = logging.getLogger('RectumSegmentationDataset')

        exluded = set()
        exclude_file =join(root_dir, 'exclude_%s.txt' % split)
        if isfile(exclude_file):
            with open(exclude_file) as exclude_fs:
                for r in exclude_fs.readlines():
                    r = r.replace('\n', '').replace('\r', '')
                    if r == '':
                        continue
                    exluded.add(r.lower())
            logger.info('Exclusion filter enabled! Read %i filenames in %s', len(exluded), exclude_file)

        image_dir = join(root_dir, split, 'image')
        target_dir = join(root_dir, split, 'label')
        self.image_filenames  = sorted([join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)])
        self.target_filenames = sorted([join(target_dir, x) for x in listdir(target_dir) if is_image_file(x)])
        assert len(self.image_filenames) == len(self.target_filenames)

        if len(exluded) > 0:
            exclude_cnt = 0
            for i in range(len(self.image_filenames) - 1, -1, -1):
                im_name = self.image_filenames[i]
                for e in exluded:
                    if e in im_name.lower():
                        del self.image_filenames[i]
                        del self.target_filenames[i]
                        exclude_cnt += 1
            logger.info('Excluded %i cases from split %s', exclude_cnt, split)
            del exclude_cnt
        del exluded
        del exclude_file

        # report the number of images in the dataset
        logger.info('Number of {0} images: {1}'.format(split, self.__len__()))

        # data augmentation
        self.transform = transform

        # data load into the ram memory
        self.preload_data = preload_data
        if self.preload_data:
            logger.info('Preloading the {0} dataset ...'.format(split))
            self.raw_images = [sitk.ReadImage(ii) for ii in self.image_filenames]
            self.raw_labels = [sitk.ReadImage(ii) for ii in self.target_filenames]
            logger.info('Loading is done\n')

    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        # load the nifti images
        if not self.preload_data:
            input = sitk.ReadImage(self.image_filenames[index])
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
