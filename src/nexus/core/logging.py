# src/ux_speech_gateway/core/logging.py
# 作用：统一日志配置。

# 内容：

# 设置日志格式、log level（读取 settings.LOG_LEVEL）。

# 可添加 JSON 日志、请求 ID、链路跟踪信息等。

# 好处：

# 所有模块调用 logging.getLogger(__name__) 都能享受统一格式和输出。

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
