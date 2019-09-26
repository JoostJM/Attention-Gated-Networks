import numpy
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import gc
import os
from functools import reduce
import operator as op

from dataio.loader import get_dataset, get_dataset_path
from dataio.transformation import get_dataset_transformation
from utils.util import json_file_to_pyobj
from utils.visualiser import Visualiser
from utils.error_logger import ErrorLogger

from models import get_model

def train(arguments):

    # Parse input arguments
    json_filename = arguments.config
    network_debug = arguments.debug

    # Load options
    json_opts = json_file_to_pyobj(json_filename)
    train_opts = json_opts.training

    # Architecture type
    arch_type = train_opts.arch_type

    # Setup Dataset and Augmentation
    ds_class = get_dataset(arch_type)
    ds_path  = get_dataset_path(arch_type, json_opts.data_path)
    ds_transform = get_dataset_transformation(arch_type, opts=json_opts.augmentation)

    # Setup the NN Model
    model = get_model(json_opts.model)
    if network_debug:
        augmentation_opts = getattr(json_opts.augmentation, arch_type)
        input_size = (train_opts.batchSize, json_opts.model.input_nc, *augmentation_opts.patch_size)
        print('# of pars: ', model.get_number_parameters())
        print('fp time: {0:.3f} sec\tbp time: {1:.3f} sec per sample'.format(*model.get_fp_bp_time(size=input_size)))
        print('Max_memory used: {0:.3f}'.format(torch.cuda.max_memory_allocated()))
        exit()

    # Setup Data Loader
    train_dataset = ds_class(ds_path, split='train',      transform=ds_transform['train'], preload_data=train_opts.preloadData)
    valid_dataset = ds_class(ds_path, split='validation', transform=ds_transform['valid'], preload_data=train_opts.preloadData)
    test_dataset  = ds_class(ds_path, split='test',       transform=ds_transform['valid'], preload_data=train_opts.preloadData)
    train_loader = DataLoader(dataset=train_dataset, num_workers=16, batch_size=train_opts.batchSize, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, num_workers=16, batch_size=train_opts.batchSize, shuffle=False)
    test_loader  = DataLoader(dataset=test_dataset,  num_workers=16, batch_size=train_opts.batchSize, shuffle=False)

    # Visualisation Parameters
    visualizer = Visualiser(json_opts.visualisation, save_dir=model.save_dir)
    error_logger = ErrorLogger()

    # Training Function
    model.set_scheduler(train_opts)
    for epoch in range(model.which_epoch, train_opts.n_epochs):
        print('(epoch: %d, total # iters: %d)' % (epoch, len(train_loader)))
        #map_memory(epoch, json_opts)

        # Training Iterations
        for epoch_iter, (images, labels) in tqdm(enumerate(train_loader, 1), total=len(train_loader)):
            # Make a training update
            model.set_input(images, labels)
            model.optimize_parameters()
            #model.optimize_parameters_accumulate_grd(epoch_iter)

            # Error visualisation
            errors = model.get_current_errors()
            error_logger.update(errors, split='train')

        # Validation and Testing Iterations
        for loader, split in zip([valid_loader, test_loader], ['validation', 'test']):
            for epoch_iter, (images, labels) in tqdm(enumerate(loader, 1), total=len(loader)):

                # Make a forward pass with the model
                model.set_input(images, labels)
                model.validate()

                # Error visualisation
                errors = model.get_current_errors()
                stats = model.get_segmentation_stats()
                error_logger.update({**errors, **stats}, split=split)

                # Visualise predictions
                visuals = model.get_current_visuals()
                visualizer.display_current_results(visuals, epoch=epoch, save_result=False)

        # Update the plots
        for split in ['train', 'validation', 'test']:
            visualizer.plot_current_errors(epoch, error_logger.get_errors(split), split_name=split)
            visualizer.print_current_errors(epoch, error_logger.get_errors(split), split_name=split)
        error_logger.reset()

        # Save the model parameters
        if epoch % train_opts.save_epoch_freq == 0:
            model.save(epoch)

        # Update the model learning rate
        model.update_learning_rate()


def map_memory(epoch, json_opts):
    with open(os.path.join(json_opts.model.checkpoints_dir, 'gc_epoch_%i.txt' % epoch), mode='w') as file_fs:
        total = 0
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    t = obj
                elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
                    t = obj
                else:
                    t = None

                if t is not None and t.is_cuda:
                    obj_size = reduce(op.mul, obj.size()) if len(obj.size()) > 0 else 0
                    total += obj_size
                    file_fs.write("%s\t%s\t%s\n" % (obj_size, type(obj), obj.size()))
            except Exception:
                pass
        file_fs.write('total: %s\n' % total)
        print('total: %s' % total)
        print(torch.cuda.memory_allocated())


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CNN Seg Training Function')

    parser.add_argument('-c', '--config',  help='training config file', required=True)
    parser.add_argument('-d', '--debug',   help='returns number of parameters and bp/fp runtime', action='store_true')
    args = parser.parse_args()

    train(args)
