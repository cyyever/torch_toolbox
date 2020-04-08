#!/usr/bin/env python3
import logging
import os

__logger_format = "%(asctime)s %(levelname)s {thd:%(thread)d} [%(filename)s => %(lineno)d] : %(message)s"
logging.basicConfig(format=__logger_format, level=logging.INFO)


def set_logger_file(filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    logger = logging.getLogger()
    handler = logging.FileHandler(filename)
    handler.setFormatter(logging.Formatter(__logger_format))
    logger.addHandler(handler)


def get_logger():
    logger = logging.getLogger()
    return logger
