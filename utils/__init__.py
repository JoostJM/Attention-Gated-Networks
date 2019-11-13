import logging.config


def configure_logging(log_level, slack=True, log_file=None):
  log_config = {
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

  if slack:
    log_config['handlers']['slack'] = {
      'class': 'utils.slack.SlackHandler',
      'bot_name': 'Attention-Gated-Networks',
      'formatter': 'std_fmt',
      'level': log_level,
    }
    log_config['loggers']['slack'] = {
      'level': log_level,
      'handlers': ['slack']
    }

  if log_file is not None:
    log_config['handlers']['file'] = {
      'class': 'logging.FileHandler',
      'formatter': 'std_fmt',
      'level': log_level,
      'filename': log_file,
      'mode': 'a'
    }
    log_config['root']['handlers'].append('file')

  logging.config.dictConfig(log_config)
