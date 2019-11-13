# Abstract level model definition
# Returns the model class for specified network type
import logging
logger = logging.getLogger('models')


def get_model(experiment, model_type, **model_opts):
    global logger

    # Neural Network Model Initialisation
    model = None

    # Print the model type
    logger.info('Initialising %s model: %s', model_type, model_opts['architecture'])

    if model_type == 'seg':
        # Return the model type
        from .feedforward_seg_model import FeedForwardSegmentation
        model = FeedForwardSegmentation(experiment, **model_opts)

    elif model_type == 'classifier':
        # Return the model type
        from .feedforward_classifier import FeedForwardClassifier
        model = FeedForwardClassifier(experiment, **model_opts)

    elif model_type == 'aggregated_classifier':
        # Return the model type
        from .aggregated_classifier import AggregatedClassifier
        model = AggregatedClassifier(experiment, **model_opts)

    return model
