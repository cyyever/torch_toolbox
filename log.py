#!/usr/bin/env python3
import logging

__logger_format = "%(asctime)s %(levelname)s {thd:%(thread)d} [%(filename)s => %(lineno)d] : %(message)s"


def set_logger_file(filename, name="ML_exp"):
    logger = logging.getLogger(name)
    logger.addHandler(logging.FileHandler(filename))


def get_logger(name="ML_exp"):
    print(__logger_format)
    logging.basicConfig(format=__logger_format)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
