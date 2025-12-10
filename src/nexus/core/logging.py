# src/ux_speech_gateway/core/logging.py
import logging
from logging.config import dictConfig


def setup_logging():
    dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'default': {
                'format': '{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
            },
        },
        'handlers': {
            'console': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
                'formatter': 'default'
            },
        },
        'loggers': {
            '': {
                'handlers': ['console'],
                'level': 'DEBUG',
                'propagate': True,
            },
        }
    })
