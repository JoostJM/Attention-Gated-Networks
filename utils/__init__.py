import logging.config
import sys

logger = logging.getLogger('utils')
_log_config = None
_log_listener = None


def configure_logging(log_level, **kwargs):
  global _log_config, _log_listener

  if _log_config is None:
    _log_config = {
      'version': 1,
      'disable_existing_loggers': False,
      'formatters': {
        'std_fmt': {
          'format': '[%(asctime)s] %(levelname)-.1s: %(message)s',
          'datefmt': '%Y-%m-%d %H:%M:%S'
        }
      },
      'handlers': {
        'console': {
          'class': 'logging.StreamHandler',
          'formatter': 'std_fmt',
          'level': log_level,
          'stream': 'ext://sys.stdout'
        }
      },
      'root': {
        'level': log_level,
        'handlers': ['console']
      },
      'loggers': {}

    }

    thread_safe = kwargs.get('thread_safe', False)
    slack = kwargs.get('slack', False)
    log_file = kwargs.get('log_file', None)

    if thread_safe:
      _log_config['formatters']['std_fmt']['format'] = \
        '[%(asctime)s] (%(threadName)s) %(levelname)-.1s: %(name)s: %(message)s'

    if slack:
      _log_config['handlers']['slack'] = {
        'class': 'utils.slack.SlackHandler',
        'bot_name': 'Attention-Gated-Networks',
        'formatter': 'std_fmt',
        'level': log_level,
      }
      _log_config['loggers']['slack'] = {
        'level': log_level,
        'handlers': ['slack']
      }

    if log_file is not None:
      py_version = (sys.version_info.major, sys.version_info.minor)
      if thread_safe and py_version >= (3, 2):
        import multiprocessing
        from logging import handlers
        log_queue = multiprocessing.Manager().Queue(-1)
        file_handler = logging.FileHandler(filename=log_file, mode='a')
        file_handler.setFormatter(logging.Formatter(fmt=_log_config['formatters']['std_fmt'].get('format'),
                                                    datefmt=_log_config['formatters']['std_fmt'].get('datefmt')))

        _log_listener = handlers.QueueListener(log_queue, file_handler)
        _log_listener.start()

        _log_config['handlers']['file'] = {
          'class': 'logging.handlers.QueueHandler',
          'queue': log_queue,
          'level': log_level,
          'formatter': 'std_fmt'
        }
      else:
        _log_config['handlers']['file'] = {
          'class': 'logging.FileHandler',
          'formatter': 'std_fmt',
          'level': log_level,
          'filename': log_file,
          'mode': 'a'
        }
      _log_config['root']['handlers'].append('file')

    logging.config.dictConfig(_log_config)
    logging.getLogger().debug("Logging configured")
  else:
    logger.warning('Logging has already been initialized')

  return _log_config, _log_listener
