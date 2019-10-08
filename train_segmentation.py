import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import gc
import os
import logging
from functools import reduce
import operator as op

from dataio.loader import get_dataset, get_dataset_path
from dataio.transformation import get_dataset_transformation
from utils.util import json_file_to_pyobj
from utils.visualiser import Visualiser
from utils.error_logger import ErrorLogger
from utils import _configure_logging

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

    experiment = json_opts.model.experiment_name

    # Set up Logging
    if not os.path.isdir(json_opts.model.checkpoints_dir):
        os.makedirs(json_opts.model.checkpoints_dir)
    log_file = os.path.join(json_opts.model.checkpoints_dir, experiment + '.log')
    _configure_logging(logging.INFO, arguments.slack, log_file)
    logger = logging.getLogger()
    slack_logger = logging.getLogger('slack')

    # Try to enable cudnn benchmark if cuda will be used
    try:
        if json_opts.model.gpu_ids is not None and len(json_opts.model.gpu_ids) > 0:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            logger.info('CuDNN benchmark enabled')
    except Exception:
        logger.warning('Failed to enable CuDNN benchmark', exc_info=True)

    # Setup Dataset and Augmentation
    ds_class = get_dataset(arch_type)
    ds_path  = get_dataset_path(arch_type, json_opts.data_path)
    ds_transform = get_dataset_transformation(arch_type, opts=json_opts.augmentation)

    # Setup the NN Model
    model = get_model(json_opts.model)
    if network_debug:
        augmentation_opts = getattr(json_opts.augmentation, arch_type)
        input_size = (train_opts.batchSize, json_opts.model.input_nc, *augmentation_opts.patch_size)
        logger.info('# of pars: %s', model.get_number_parameters())
        logger.info('fp time: {0:.3f} sec\tbp time: {1:.3f} sec per sample'.format(*model.get_fp_bp_time(size=input_size)))
        logger.info('Max_memory used: {0:.3f}'.format(torch.cuda.max_memory_allocated()))
        exit()

    # Setup Data Loader
    train_dataset = ds_class(ds_path, split='train',      transform=ds_transform['train'], preload_data=train_opts.preloadData)
    valid_dataset = ds_class(ds_path, split='validation', transform=ds_transform['valid'], preload_data=train_opts.preloadData)
    train_loader = DataLoader(dataset=train_dataset, num_workers=4, batch_size=train_opts.batchSize, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, num_workers=4, batch_size=train_opts.batchSize, shuffle=False)

    # Visualisation Parameters
    visualizer = Visualiser(json_opts.visualisation, save_dir=model.save_dir)
    error_logger = ErrorLogger()

    # Training Function
    model.set_scheduler(train_opts)
    slack_logger.info('Starting training for experiment %s', json_opts.model.experiment_name)
    try:
        accumulate_iter = getattr(train_opts, "accumulate_iter", 1)
        if accumulate_iter > 1:
          logger.info('Accumulating gradients every %d iters', accumulate_iter)

        assert model.which_epoch < train_opts.n_epochs, \
            'Model training already at designated number of epochs (%i)' % train_opts.n_epochs
        for epoch in range(model.which_epoch + 1, train_opts.n_epochs + 1):
            logger.info('(epoch: %d, total # iters: %d)' % (epoch, len(train_loader)))

            # Training Iterations
            for epoch_iter, (images, labels) in tqdm(enumerate(train_loader, 1), total=len(train_loader)):
                # Make a training update
                model.set_input(images, labels)
                model.optimize_parameters(epoch_iter, accumulate_iter)

                # Error visualisation
                errors = model.get_current_errors()
                error_logger.update(errors, split='train')
                del images
                del labels

            # Update the network parameters if some have been accumulated (epoch_iter % accumulate_iter != 0)
            # Reflects update from different-sized final batch
            if epoch_iter % accumulate_iter != 0:
                model.optimizer_S.step()

            # Validation Iterations
            for epoch_iter, (images, labels) in tqdm(enumerate(valid_loader, 1), total=len(valid_loader)):
                # Make a forward pass with the model
                if hasattr(torch, 'no_grad'):
                    with torch.no_grad:
                        model.set_input(images, labels)
                        model.validate()
                else:
                    model.set_input(images, labels)
                    model.validate()

                # Error visualisation
                errors = model.get_current_errors()
                stats = model.get_segmentation_stats()
                error_logger.update({**errors, **stats}, split='validation')

                # Visualise predictions
                visuals = model.get_current_visuals()
                visualizer.display_current_results(visuals, epoch=epoch, save_result=False)
                del images
                del labels

            # Update the plots
            for split in error_logger.variables.keys():
                visualizer.plot_current_errors(epoch, error_logger.get_errors(split), split_name=split)
                visualizer.print_current_errors(epoch, error_logger.get_errors(split), split_name=split)

            # Save the model parameters
            if epoch % train_opts.save_epoch_freq == 0:
                slack_logger.info('(Experiment %s) Saving model at epoch %04d, loss:%s',
                                  json_opts.model.experiment_name, epoch, _get_loss_msg(error_logger))

                model.save(epoch)

            error_logger.reset()

            # Update the model learning rate
            model.update_learning_rate()

            if model.use_cuda:
                torch.cuda.empty_cache()

        # Store the final model
        slack_logger.info('(Experiment %s) Training finished! Saving model...',
                          json_opts.model.experiment_name)
        model.save(epoch)

        if arguments.eval:
            import eval_segmentation
            eval_segmentation.eval(model, json_opts)
    except Exception:
        slack_logger.critical('(Experiment %s) Oh No! Training failed!!', json_opts.model.experiment_name, exc_info=True)


def _get_loss_msg(error_logger):
    loss_msg = ''
    for split in error_logger.variables.keys():
        loss_msg += '\n\t (split %s)' % split
        for k, v in error_logger.get_errors(split).items():
            if np.isscalar(v):
                loss_msg += '%s: %.3f ' % (k, v)
    return loss_msg


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CNN Seg Training Function')

    parser.add_argument('-c', '--config',  help='training config file', required=True)
    parser.add_argument('-d', '--debug',   help='returns number of parameters and bp/fp runtime', action='store_true')
    parser.add_argument('-s', '--slack',   help='enables logging to Slack Messenger', action='store_true')
    parser.add_argument('-e', '--eval',    help='enables creating evaluation of the final model', action='store_true')

    args = parser.parse_args()

    train(args)
