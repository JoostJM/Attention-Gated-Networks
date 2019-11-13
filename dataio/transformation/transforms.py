import inspect
import logging

import SimpleITK as sitk
import six
import torchsample.transforms as ts


class Transformations:

  def __init__(self, name, **transform_opts):
    self.logger = logging.getLogger(self.__module__)

    def is_transform(mem):
      if not inspect.ismethod(mem):
        return False
      return mem.__name__.endswith('_transform')

    transforms = {k: v for k, v in inspect.getmembers(self, is_transform)}
    assert name in transforms, "Transform %s not recognized, choose from %s" % (name, transforms)
    self.transform = transforms[name]

    self.transform_opts = {
      # Input patch and scale size
      'scale_size': (192, 192, 1),
      'patch_size': (128, 128, 1),

      # Affine and Intensity Transformations
      'shift_val': 0.1,
      'rotate_val': 15.0,
      'scale_val': (0.7, 1.3),
      'inten_val': (1.0, 1.0),
      'random_flip_prob': 0.0,

      # Divisibility factor for testing
      'division_factor': (16, 16, 1)
    }
    self.transform_opts.update(transform_opts)

    pretty_opts = ''
    for k, v in six.iteritems(self.transform_opts):
      pretty_opts += '\n\t%s: %s' % (k, v)

    self.logger.info('Initialized augmentation %s with options:%s', name, pretty_opts)

  def ukbb_sax_transform(self):
    train_transform = ts.Compose([ts.PadNumpy(size=self.transform_opts['scale_size']),
                                  ts.ToTensor(),
                                  ts.ChannelsFirst(),
                                  ts.TypeCast(['float', 'float']),
                                  ts.RandomFlip(h=True, v=True, p=self.transform_opts['random_flip_prob']),
                                  ts.RandomAffine(rotation_range=self.transform_opts['rotate_val'],
                                                  translation_range=self.transform_opts['shift_val'],
                                                  zoom_range=self.transform_opts['scale_val'],
                                                  interp=('bilinear', 'nearest')),
                                  ts.NormalizeMedicPercentile(norm_flag=(True, False)),
                                  ts.RandomCrop(size=self.transform_opts['patch_size']),
                                  ts.TypeCast(['float', 'long'])
                                  ])

    valid_transform = ts.Compose([ts.PadNumpy(size=self.transform_opts['scale_size']),
                                  ts.ToTensor(),
                                  ts.ChannelsFirst(),
                                  ts.TypeCast(['float', 'float']),
                                  ts.NormalizeMedicPercentile(norm_flag=(True, False)),
                                  ts.SpecialCrop(size=self.transform_opts['patch_size'], crop_type=0),
                                  ts.TypeCast(['float', 'long'])
                                  ])

    return {'train': train_transform, 'valid': valid_transform}

  def cmr_3d_sax_transform(self):
    train_transform = ts.Compose([ts.PadNumpy(size=self.transform_opts['scale_size'], channels_first=False),
                                  ts.ToTensor(),
                                  ts.ChannelsFirst(),
                                  ts.TypeCast(['float', 'float']),
                                  ts.RandomFlip(h=True, v=True, p=self.transform_opts['random_flip_prob']),
                                  ts.RandomAffine3D(rotation_range=self.transform_opts['rotate_val'],
                                                    translation_range=self.transform_opts['shift_val'],
                                                    zoom_range=self.transform_opts['scale_val'],
                                                    interp=('trilinear', 'nearest')),
                                  # ts.NormalizeMedicPercentile(norm_flag=(True, False)),
                                  ts.NormalizeMedic(norm_flag=(True, False)),
                                  ts.RandomCrop(size=self.transform_opts['patch_size'], channels_first=True),
                                  ts.TypeCast(['float', 'long'])
                                  ])

    valid_transform = ts.Compose([ts.PadNumpy(size=self.transform_opts['scale_size'], channels_first=False),
                                  ts.ToTensor(),
                                  ts.ChannelsFirst(),
                                  ts.TypeCast(['float', 'float']),
                                  # ts.NormalizeMedicPercentile(norm_flag=(True, False)),
                                  ts.NormalizeMedic(norm_flag=(True, False)),
                                  ts.SpecialCrop(size=self.transform_opts['patch_size'],
                                                 crop_type=0, channels_first=True),
                                  ts.TypeCast(['float', 'long'])
                                  ])

    return {'train': train_transform, 'valid': valid_transform}

  def cmr_3d_sitk_transform(self):
    train_transform = ts.Compose([ts.PadSimpleITK(size=self.transform_opts['scale_size']),
                                  ts.RandomFlipSimpleITK(h=True, v=True, p=self.transform_opts['random_flip_prob']),
                                  ts.RandomAffineSimpleITK(rotation_range=self.transform_opts['rotate_val'],
                                                           translation_range=self.transform_opts['shift_val'],
                                                           zoom_range=self.transform_opts['scale_val'],
                                                           interp=(sitk.sitkLinear, sitk.sitkNearestNeighbor)),
                                  ts.NormalizeSimpleITK(norm_flag=[True, False]),
                                  ts.RandomCropSimpleITK(size=self.transform_opts['patch_size']),
                                  ts.SimpleITKtoTensor(),
                                  ts.ChannelsFirst(),
                                  ts.TypeCast(['float', 'long'])
                                  ])

    valid_transform = ts.Compose([ts.PadSimpleITK(size=self.transform_opts['scale_size']),
                                  ts.NormalizeSimpleITK(norm_flag=[True, False]),
                                  ts.SpecialCropSimpleITK(size=self.transform_opts['patch_size'], crop_type=0),
                                  ts.SimpleITKtoTensor(),
                                  ts.ChannelsFirst(),
                                  ts.TypeCast(['float', 'long'])
                                  ])

    return {'train': train_transform, 'valid': valid_transform}

  def hms_sax_transform(self):
    # Training transformation
    # 2D Stack input - 3D High Resolution output segmentation

    train_transform = []
    valid_transform = []

    # First pad to a fixed size
    # Torch tensor
    # Channels first
    # Joint affine transformation
    # In-plane respiratory motion artefacts (translation and rotation)
    # Random Crop
    # Normalise the intensity range
    train_transform = ts.Compose([])

    return {'train': train_transform, 'valid': valid_transform}

  def test_3d_sax_transform(self):
    test_transform = ts.Compose([ts.PadFactorNumpy(factor=self.transform_opts['division_factor']),
                                 ts.ToTensor(),
                                 ts.ChannelsFirst(),
                                 ts.TypeCast(['float']),
                                 # ts.NormalizeMedicPercentile(norm_flag=True),
                                 ts.NormalizeMedic(norm_flag=True),
                                 ts.ChannelsLast(),
                                 ts.AddChannel(axis=0),
                                 ])

    return {'test': test_transform}

  def ultrasound_transform(self):
    train_transform = ts.Compose([ts.ToTensor(),
                                  ts.TypeCast(['float']),
                                  ts.AddChannel(axis=0),
                                  ts.SpecialCrop(self.transform_opts['patch_size'], 0),
                                  ts.RandomFlip(h=True, v=False, p=self.transform_opts['random_flip_prob']),
                                  ts.RandomAffine(rotation_range=self.transform_opts['rotate_val'],
                                                  translation_range=self.transform_opts['shift_val'],
                                                  zoom_range=self.transform_opts['scale_val'],
                                                  interp=('bilinear')),
                                  ts.StdNormalize(),
                                  ])

    valid_transform = ts.Compose([ts.ToTensor(),
                                  ts.TypeCast(['float']),
                                  ts.AddChannel(axis=0),
                                  ts.SpecialCrop(self.transform_opts['patch_size'], 0),
                                  ts.StdNormalize(),
                                  ])

    return {'train': train_transform, 'valid': valid_transform}
