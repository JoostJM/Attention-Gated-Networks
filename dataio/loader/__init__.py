from dataio.loader.ukbb_dataset import UKBBDataset
from dataio.loader.test_dataset import TestDataset
from dataio.loader.hms_dataset import HMSDataset
from dataio.loader.cmr_3D_dataset import CMR3DDataset
from dataio.loader.us_dataset import UltraSoundDataset
from dataio.loader.rectum_segmentation_dataset import RectumSegmentationDataset

from dataio.transformation import Transformations

_datasets = {str(c.__name__): c for c in
             (UKBBDataset, TestDataset, HMSDataset, CMR3DDataset, UltraSoundDataset, RectumSegmentationDataset)}


def get_dataset(splits, dataset_class, data_path, augmentation=None, **data_opts):
    """get_dataset

    :param splits: Splits to return a dataset object for
    :param dataset_class: Name of the class to use for loading the dataset
    :param data_path: Location of the dataset to load
    :param augmentation: Optional dictionary specifying the data augmentation configuration
    :param data_opts: Additional keywords used at initialization of the dataset_class

    :returns: Dictionary with splits as keys and dataset_class objects as values
    """
    global _datasets

    data_opts = data_opts.copy()

    cls = _datasets[dataset_class]
    if augmentation is not None:
      transforms = Transformations(**augmentation).transform()
    else:
      transforms = None

    datasets = {}
    for split in splits:
      tf = None
      if transforms is not None:
        tf = transforms['train' if split == 'train' else 'valid']

      datasets[split] = cls(data_path, split=split, transform=tf, **data_opts)

    return datasets
